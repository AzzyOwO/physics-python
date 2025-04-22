import pygame
import sys
import numpy as np
import random
from datetime import datetime

# Initialize pygame
pygame.init()

# Screen dimensions
WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Physics Emulator")

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
GREEN = (0, 255, 0)
YELLOW = (255, 255, 0)

# Physics parameters
GRAVITY = np.array([0, 0.5], dtype=np.float64)
FRICTION = 0.99
ELASTICITY = 0.8

class Particle:
    def __init__(self, x, y, radius=10, mass=1):
        self.pos = np.array([x, y], dtype=np.float64)
        self.vel = np.array([0, 0], dtype=np.float64)
        self.acc = np.array([0, 0], dtype=np.float64)
        self.radius = radius
        self.mass = mass
        self.color = (random.randint(50, 255), random.randint(50, 255), random.randint(50, 255))
        self.fixed = False
        self.id = random.randint(1000, 9999)

    def apply_force(self, force):
        self.acc += force / self.mass

    def update(self):
        if not self.fixed:
            self.vel += self.acc
            self.vel *= FRICTION
            self.pos += self.vel
            self.acc = np.array([0, 0], dtype=np.float64)

        # Boundary collisions
        if self.pos[0] <= self.radius:
            self.pos[0] = self.radius
            self.vel[0] *= -ELASTICITY
        elif self.pos[0] >= WIDTH - self.radius:
            self.pos[0] = WIDTH - self.radius
            self.vel[0] *= -ELASTICITY

        if self.pos[1] <= self.radius:
            self.pos[1] = self.radius
            self.vel[1] *= -ELASTICITY
        elif self.pos[1] >= HEIGHT - self.radius:
            self.pos[1] = HEIGHT - self.radius
            self.vel[1] *= -ELASTICITY

    def draw(self, screen):
        pygame.draw.circle(screen, self.color, self.pos.astype(int), self.radius)
        if debug_mode:
            end_pos = self.pos + self.vel * 10
            pygame.draw.line(screen, GREEN, self.pos, end_pos, 2)
            end_acc = self.pos + self.acc * 100
            pygame.draw.line(screen, RED, self.pos, end_acc, 2)

def check_collision(p1, p2):
    distance = np.linalg.norm(p1.pos - p2.pos)
    return distance < (p1.radius + p2.radius)

def resolve_collision(p1, p2):
    if p1.fixed and p2.fixed:
        return

    normal = p2.pos - p1.pos
    distance = np.linalg.norm(normal)
    normal = normal / distance if distance > 0 else np.array([1, 0], dtype=np.float64)

    mtd = normal * (p1.radius + p2.radius - distance)

    im1 = 1 / p1.mass if not p1.fixed else 0
    im2 = 1 / p2.mass if not p2.fixed else 0

    p1.pos -= mtd * (im1 / (im1 + im2))
    p2.pos += mtd * (im2 / (im1 + im2))

    relative_vel = p2.vel - p1.vel
    velocity_along_normal = np.dot(relative_vel, normal)

    if velocity_along_normal > 0:
        return

    j = -(1 + ELASTICITY) * velocity_along_normal
    j /= (im1 + im2)

    impulse = normal * j
    p1.vel -= impulse * im1
    p2.vel += impulse * im2

def main():
    global debug_mode
    debug_mode = False
    secret_code = "&RN%/K"
    entered_code = ""
    last_key_time = datetime.now()

    clock = pygame.time.Clock()
    particles = []
    selected_particle = None
    paused = False

    target_fps = 60
    min_fps = 15
    max_fps = 720

    fps = 0
    tps = 0
    fps_counter = 0
    tps_counter = 0
    fps_timer = 0
    tps_timer = 0

    for _ in range(10):
        x = random.randint(50, WIDTH - 50)
        y = random.randint(50, HEIGHT - 50)
        radius = random.randint(10, 20)
        mass = radius ** 2
        particles.append(Particle(x, y, radius, mass))

    fixed_particle = Particle(WIDTH // 2, HEIGHT // 4, 15, 1000)
    fixed_particle.fixed = True
    fixed_particle.color = RED
    particles.append(fixed_particle)

    while True:
        current_time = datetime.now()
        delta_time = clock.tick() / 1000.0 if target_fps == 0 else clock.tick(target_fps) / 1000.0

        fps_counter += 1
        fps_timer += delta_time
        if fps_timer >= 1.0:
            fps = fps_counter
            fps_counter = 0
            fps_timer -= 1.0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key == pygame.K_c:
                    particles = [p for p in particles if p.fixed]
                elif event.key == pygame.K_r:
                    particles = [p for p in particles if p.fixed]
                    for _ in range(10):
                        x = random.randint(50, WIDTH - 50)
                        y = random.randint(50, HEIGHT - 50)
                        radius = random.randint(10, 20)
                        mass = radius ** 2
                        particles.append(Particle(x, y, radius, mass))
                elif event.key == pygame.K_d:
                    debug_mode = not debug_mode
                elif event.key == pygame.K_UP:
                    if target_fps < max_fps:
                        target_fps += 15
                elif event.key == pygame.K_DOWN:
                    if target_fps > min_fps:
                        target_fps -= 15
                elif event.key == pygame.K_u:
                    target_fps = 0 if target_fps != 0 else 60
                else:
                    now = datetime.now()
                    if (now - last_key_time).total_seconds() > 2.0:
                        entered_code = ""
                    last_key_time = now
                    if event.key == pygame.K_AMPERSAND:
                        entered_code += "&"
                    elif event.key == pygame.K_r:
                        entered_code += "R"
                    elif event.key == pygame.K_n:
                        entered_code += "N"
                    elif event.key == pygame.K_PERCENT:
                        entered_code += "%"
                    elif event.key == pygame.K_SLASH:
                        entered_code += "/"
                    elif event.key == pygame.K_k:
                        entered_code += "K"
                    if entered_code == secret_code:
                        debug_mode = not debug_mode
                        entered_code = ""

            elif event.type == pygame.MOUSEBUTTONDOWN:
                mouse_pos = np.array(pygame.mouse.get_pos(), dtype=np.float64)
                if event.button == 1:
                    for p in particles:
                        if np.linalg.norm(p.pos - mouse_pos) < p.radius:
                            selected_particle = p
                            break
                    else:
                        radius = random.randint(5, 15)
                        mass = radius ** 2
                        new_particle = Particle(*mouse_pos, radius, mass)
                        particles.append(new_particle)
                elif event.button == 3:
                    for p in particles:
                        if np.linalg.norm(p.pos - mouse_pos) < p.radius:
                            p.fixed = not p.fixed
                            p.color = RED if p.fixed else p.color
                            break
            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:
                    selected_particle = None
            elif event.type == pygame.MOUSEMOTION and selected_particle:
                if not selected_particle.fixed:
                    selected_particle.pos = np.array(pygame.mouse.get_pos(), dtype=np.float64)
                    selected_particle.vel = np.array([0, 0], dtype=np.float64)

        if not paused:
            tps_counter += 1
            tps_timer += delta_time
            if tps_timer >= 1.0:
                tps = tps_counter
                tps_counter = 0
                tps_timer -= 1.0

            for p in particles:
                if not p.fixed:
                    p.apply_force(GRAVITY * p.mass)

            for p in particles:
                p.update()

            for i in range(len(particles)):
                for j in range(i + 1, len(particles)):
                    if check_collision(particles[i], particles[j]):
                        resolve_collision(particles[i], particles[j])

        screen.fill(BLACK)

        font = pygame.font.SysFont(None, 24)
        instructions = [
            "Left click: Select/Create particle",
            "Right click: Toggle fixed position",
            "Space: Pause/Resume",
            "C: Clear particles",
            "R: Reset with new particles",
            "D: Toggle debug mode",
            "Up/Down: Adjust FPS cap",
            "U: Unlock/lock FPS"
        ]

        for i, text in enumerate(instructions):
            text_surface = font.render(text, True, WHITE)
            screen.blit(text_surface, (10, 10 + i * 25))

        fps_cap_status = "Unlocked" if target_fps == 0 else f"{target_fps}"
        metrics = [
            f"FPS: {fps}",
            f"TPS: {tps if not paused else 0}",
            f"Particles: {len(particles)}",
            f"Debug: {'ON' if debug_mode else 'OFF'}",
            f"FPS Cap: {fps_cap_status}"
        ]

        for i, text in enumerate(metrics):
            text_surface = font.render(text, True, YELLOW)
            screen.blit(text_surface, (WIDTH - 180, 10 + i * 25))

        for p in particles:
            p.draw(screen)
            if debug_mode:
                debug_text = f"ID: {p.id} M: {p.mass:.1f}"
                text_surface = font.render(debug_text, True, WHITE)
                screen.blit(text_surface, (p.pos[0] + p.radius + 5, p.pos[1] - 10))

        pygame.display.flip()

if __name__ == "__main__":
    main()
