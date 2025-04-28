import pygame
import sys
import math
import random
import numpy as np
from datetime import datetime

# --- Initialization ---
pygame.init()

INIT_WIDTH, INIT_HEIGHT = 1200, 720
screen = pygame.display.set_mode((INIT_WIDTH, INIT_HEIGHT), pygame.RESIZABLE)
pygame.display.set_caption("Physics Sim - Ultimate Blend")

# --- Colors ---
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GRAY = (200, 200, 200)
DARK_GRAY = (30, 30, 40)
LIGHT_GRAY = (60, 60, 80)
COLORS = [(255, 80, 80), (0, 255, 0), (0, 150, 255), (255, 255, 0), (0, 255, 255), (200, 200, 255)]

# --- Fonts ---
font = pygame.font.SysFont("Consolas", 18)
title_font = pygame.font.SysFont("Consolas", 32, bold=True)

clock = pygame.time.Clock()

# --- Physics ---
GRAVITY = np.array([0, 0.5], dtype=np.float64)
FRICTION = 0.99
ELASTICITY = 0.8

# --- Settings ---
target_fps = 60
min_fps = 15
max_fps = 720
fps_locked = True
show_tab = True

# --- Classes ---
class Particle:
    def __init__(self, x, y, radius=None, mass=None):
        self.pos = np.array([x, y], dtype=np.float64)
        self.vel = np.array([0, 0], dtype=np.float64)
        self.acc = np.array([0, 0], dtype=np.float64)
        self.radius = radius if radius else random.randint(5, 15)
        self.mass = mass if mass else self.radius ** 2
        self.fixed = False
        self.color = random.choice(COLORS)
        self.id = random.randint(1000, 9999)

    def apply_force(self, force):
        if not self.fixed:
            self.acc += force / self.mass

    def update(self, width, height):
        if not self.fixed:
            self.vel += self.acc
            self.vel *= FRICTION
            self.pos += self.vel
            self.acc = np.array([0, 0], dtype=np.float64)

        # Boundary collisions
        if self.pos[0] <= self.radius:
            self.pos[0] = self.radius
            self.vel[0] *= -ELASTICITY
        elif self.pos[0] >= width - self.radius:
            self.pos[0] = width - self.radius
            self.vel[0] *= -ELASTICITY

        if self.pos[1] <= self.radius:
            self.pos[1] = self.radius
            self.vel[1] *= -ELASTICITY
        elif self.pos[1] >= height - self.radius:
            self.pos[1] = height - self.radius
            self.vel[1] *= -ELASTICITY

    def draw(self, screen, debug_mode, selected_particle):
        if selected_particle and selected_particle.id == self.id:
            pygame.draw.circle(screen, (255, 255, 0), self.pos.astype(int), self.radius + 3, 2)

        if self.fixed:
            pulse = int(3 * abs(np.sin(pygame.time.get_ticks() / 300)))
            pygame.draw.circle(screen, (255, 100, 100), self.pos.astype(int), self.radius + pulse)

        pygame.draw.circle(screen, self.color, self.pos.astype(int), self.radius)

        if debug_mode:
            vel_line = self.pos + self.vel * 10
            pygame.draw.line(screen, (0, 255, 0), self.pos, vel_line, 2)
            debug_text = font.render(f"ID:{self.id} M:{self.mass:.1f}", True, (255, 255, 255))
            screen.blit(debug_text, (self.pos[0] + self.radius + 5, self.pos[1] - 10))

class Constraint:
    def __init__(self, p1, p2):
        self.p1 = p1
        self.p2 = p2
        self.length = np.linalg.norm(p1.pos - p2.pos)

    def update(self):
        delta = self.p2.pos - self.p1.pos
        dist = np.linalg.norm(delta)
        if dist == 0:
            return
        diff = (self.length - dist) / dist / 2
        offset = delta * diff
        if not self.p1.fixed:
            self.p1.pos -= offset
        if not self.p2.fixed:
            self.p2.pos += offset

    def draw(self, screen):
        pygame.draw.line(screen, GRAY, self.p1.pos.astype(int), self.p2.pos.astype(int), 2)

# --- Collision Detection ---
def resolve_collision(p1, p2):
    if p1.fixed and p2.fixed:
        return

    normal = p2.pos - p1.pos
    distance = np.linalg.norm(normal)
    if distance == 0:
        return
    normal = normal / distance

    overlap = (p1.radius + p2.radius) - distance
    if overlap > 0:
        mtd = normal * overlap
        im1 = 1 / p1.mass if not p1.fixed else 0
        im2 = 1 / p2.mass if not p2.fixed else 0
        p1.pos -= mtd * (im1 / (im1 + im2))
        p2.pos += mtd * (im2 / (im1 + im2))

        relative_vel = p2.vel - p1.vel
        vel_along_normal = np.dot(relative_vel, normal)

        if vel_along_normal > 0:
            return

        j = -(1 + ELASTICITY) * vel_along_normal
        j /= (im1 + im2)

        impulse = normal * j
        if not p1.fixed:
            p1.vel -= impulse * im1
        if not p2.fixed:
            p2.vel += impulse * im2

# --- Game Variables ---
particles = []
constraints = []
selected_particle = None
paused = False
debug_mode = False

# --- Main Loop ---
running = True
while running:
    current_width, current_height = screen.get_size()
    delta_time = clock.tick(target_fps) / 1000.0 if fps_locked else clock.tick() / 1000.0

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_TAB:
                show_tab = not show_tab
            if event.key == pygame.K_SPACE:
                paused = not paused
            elif event.key == pygame.K_c:
                particles.clear()
                constraints.clear()
            elif event.key == pygame.K_r:
                for p in particles:
                    p.vel = np.array([0, 0], dtype=np.float64)
            elif event.key == pygame.K_d:
                debug_mode = not debug_mode
            elif event.key == pygame.K_UP:
                target_fps = min(max_fps, target_fps + 15)
            elif event.key == pygame.K_DOWN:
                target_fps = max(min_fps, target_fps - 15)
            elif event.key == pygame.K_u:
                fps_locked = not fps_locked
            elif event.key == pygame.K_f:
                pygame.display.toggle_fullscreen()

        elif event.type == pygame.MOUSEBUTTONDOWN:
            mouse_pos = np.array(pygame.mouse.get_pos(), dtype=np.float64)
            if event.button == 1:
                for p in particles:
                    if np.linalg.norm(p.pos - mouse_pos) < p.radius + 5:
                        selected_particle = p
                        break
                else:
                    p = Particle(mouse_pos[0], mouse_pos[1])
                    particles.append(p)

            elif event.button == 3:
                for p in particles:
                    if np.linalg.norm(p.pos - mouse_pos) < p.radius + 5:
                        p.fixed = not p.fixed
                        break

            elif event.button == 2:
                for p in particles:
                    if np.linalg.norm(p.pos - mouse_pos) < p.radius + 5:
                        p.fixed = not p.fixed
                        break

        elif event.type == pygame.MOUSEBUTTONUP:
            if event.button == 1:
                selected_particle = None

    if not paused:
        for _ in range(5):
            for c in constraints:
                c.update()
        for i in range(len(particles)):
            for j in range(i + 1, len(particles)):
                resolve_collision(particles[i], particles[j])
        for p in particles:
            p.apply_force(GRAVITY * p.mass)
            p.update(current_width, current_height)

    screen.fill(BLACK)

    for c in constraints:
        c.draw(screen)

    for p in particles:
        p.draw(screen, debug_mode, selected_particle)

    if show_tab:
        controls = [
            "Left Click: Select/Create particle",
            "Middle Click: Toggle Fixed",
            "Right Click: Freeze/Unfreeze particle",
            "Space: Pause/Resume",
            "C: Clear All",
            "R: Reset Velocities",
            "D: Toggle Debug",
            "UP/DOWN: FPS + -",
            "U: Toggle FPS Lock",
            "F: Toggle Fullscreen",
            "TAB: Toggle This Tab"
        ]
        for i, line in enumerate(controls):
            text = font.render(line, True, WHITE)
            screen.blit(text, (10, 70 + i * 24))
    else:
        fps_text = font.render(f"FPS: {int(clock.get_fps())}", True, (255, 255, 0))
        screen.blit(fps_text, (10, 10))

    screen.blit(title_font.render("Physics Sim", True, (200, 200, 255)), (current_width//2 - 120, 10))

    pygame.display.flip()

pygame.quit()
sys.exit()

