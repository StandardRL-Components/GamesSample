import gymnasium as gym
import os
import pygame
import pygame.gfxdraw
import math
import random
import numpy as np
from gymnasium.spaces import MultiDiscrete, Box
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    """
    Gymnasium environment where the player guides a splitting particle to a goal.

    The player controls the velocity of the main particle and can transform it
    into a wave. Colliding with obstacles splits the particle into two smaller,
    faster children. The goal is to reach the target before the timer runs out.

    **Visuals:**
    - Minimalist, geometric style with glowing effects.
    - Player Particle: Bright blue, controllable.
    - Child Particles: Smaller, inherit color.
    - Wave Form: A cyan sine wave.
    - Obstacles: Red, static rectangles.
    - Goal: A green, glowing circle.

    **Gameplay:**
    - Use arrow keys to apply force to the main particle.
    - Press Space to toggle between particle and wave form.
    - Wave form follows a sinusoidal path, useful for navigating tight spaces.
    - Hitting an obstacle splits your particle, creating more entities to manage.
    - Only the main particle can be controlled.
    - Reach the goal with any particle to win.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Guide a particle to a goal, splitting on obstacles and transforming into a wave to navigate. "
        "Reach the goal with any particle before time runs out."
    )
    user_guide = "Use arrow keys (↑↓←→) to apply force to your particle. Press space to toggle between particle and wave form."
    auto_advance = True

    # --- Constants ---
    WIDTH, HEIGHT = 640, 400
    FPS = 30
    MAX_TIME_SECONDS = 45
    MAX_STEPS = int(MAX_TIME_SECONDS * FPS)

    # Colors
    COLOR_BG = (15, 15, 30)
    COLOR_PARTICLE = (0, 150, 255)
    COLOR_WAVE = (0, 255, 255)
    COLOR_OBSTACLE = (255, 50, 50)
    COLOR_GOAL = (50, 255, 50)
    COLOR_TEXT = (220, 220, 220)
    COLOR_VECTOR = (100, 100, 100)

    # Physics
    PARTICLE_ACCEL = 0.5
    PARTICLE_MAX_SPEED = 4.0
    PARTICLE_DRAG = 0.99
    MIN_SPLIT_RADIUS = 2

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        self.render_mode = render_mode

        # --- Gymnasium Spaces ---
        self.observation_space = Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_main = pygame.font.SysFont("monospace", 20, bold=True)
        self.font_small = pygame.font.SysFont("monospace", 14)

        # --- Game State Variables ---
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.timer = 0.0
        self.particles = []
        self.obstacles = []
        self.goal_pos = pygame.math.Vector2(0, 0)
        self.goal_radius = 0
        self.space_was_held = False
        
        # self.reset() # reset is called by the wrapper/user

    def _create_particle(self, pos, vel, radius, parent_id=-1, is_main=False):
        return {
            "id": random.randint(1000, 9999),
            "parent_id": parent_id,
            "pos": pygame.math.Vector2(pos),
            "vel": pygame.math.Vector2(vel),
            "radius": radius,
            "is_wave": False,
            "wave_phase": random.uniform(0, 2 * math.pi),
            "is_main": is_main,
            "color": self.COLOR_PARTICLE,
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.timer = self.MAX_TIME_SECONDS
        self.space_was_held = False

        # --- Place Goal ---
        self.goal_radius = 20
        self.goal_pos = pygame.math.Vector2(
            self.WIDTH - 50, self.HEIGHT / 2
        )

        # --- Place Player Particle ---
        player_start_pos = pygame.math.Vector2(50, self.HEIGHT / 2)
        self.particles = [
            self._create_particle(
                pos=player_start_pos,
                vel=(0, 0),
                radius=8,
                is_main=True
            )
        ]

        # --- Place Obstacles ---
        self.obstacles = []
        num_obstacles = self.np_random.integers(3, 7)
        for _ in range(num_obstacles):
            while True:
                w, h = self.np_random.integers(20, 60, size=2)
                x = self.np_random.uniform(self.WIDTH * 0.2, self.WIDTH * 0.8)
                y = self.np_random.uniform(0, self.HEIGHT - h)
                obstacle_rect = pygame.Rect(x, y, w, h)
                
                # Ensure obstacle doesn't block start or end points
                if not obstacle_rect.collidepoint(player_start_pos) and \
                   not obstacle_rect.collidepoint(self.goal_pos) and \
                   obstacle_rect.colliderect(pygame.Rect(self.goal_pos.x - self.goal_radius, self.goal_pos.y - self.goal_radius, self.goal_radius*2, self.goal_radius*2)) == 0:
                    self.obstacles.append(obstacle_rect)
                    break
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        self.steps += 1
        self.timer -= 1.0 / self.FPS
        
        main_particle = self._get_main_particle()
        dist_before = float('inf')
        if main_particle:
            dist_before = main_particle["pos"].distance_to(self.goal_pos)

        # --- Handle Actions ---
        self._handle_input(action)

        # --- Update Game State ---
        newly_created_particles = []
        particles_to_remove = []

        for p in self.particles:
            # Update position
            self._update_particle_position(p)

            # Check boundaries
            if not self.screen.get_rect().collidepoint(p["pos"]):
                particles_to_remove.append(p)
                continue
            
            # Check collisions
            new_children = self._check_collisions(p)
            if new_children:
                newly_created_particles.extend(new_children)
                particles_to_remove.append(p)

        # Update particle list
        self.particles = [p for p in self.particles if p not in particles_to_remove]
        self.particles.extend(newly_created_particles)

        # --- Calculate Reward ---
        reward = 0
        main_particle = self._get_main_particle()
        if main_particle:
            dist_after = main_particle["pos"].distance_to(self.goal_pos)
            reward += (dist_before - dist_after) * 0.1 # Distance reward
        
        # --- Check Termination ---
        terminated = self.game_over or self.timer <= 0 or self.steps >= self.MAX_STEPS
        truncated = False # Not using truncation
        if self.game_over and self.score > 0: # Reached goal
            reward = 100.0
        elif self.timer <= 0 or self.steps >= self.MAX_STEPS: # Timed out
            reward = -100.0
            self.game_over = True
            terminated = True
            
        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _get_main_particle(self):
        for p in self.particles:
            if p["is_main"]:
                return p
        return None

    def _handle_input(self, action):
        movement, space_held, _ = action[0], action[1] == 1, action[2] == 1
        
        main_particle = self._get_main_particle()
        if not main_particle:
            return

        # Movement
        if movement == 1: main_particle["vel"].y -= self.PARTICLE_ACCEL # Up
        elif movement == 2: main_particle["vel"].y += self.PARTICLE_ACCEL # Down
        elif movement == 3: main_particle["vel"].x -= self.PARTICLE_ACCEL # Left
        elif movement == 4: main_particle["vel"].x += self.PARTICLE_ACCEL # Right

        # Toggle wave on space press (rising edge detection)
        if space_held and not self.space_was_held:
            main_particle["is_wave"] = not main_particle["is_wave"]
        self.space_was_held = space_held

    def _update_particle_position(self, p):
        # Apply drag
        p["vel"] *= self.PARTICLE_DRAG
        
        # Clamp velocity
        p["vel"].x = np.clip(p["vel"].x, -self.PARTICLE_MAX_SPEED, self.PARTICLE_MAX_SPEED)
        p["vel"].y = np.clip(p["vel"].y, -self.PARTICLE_MAX_SPEED, self.PARTICLE_MAX_SPEED)

        # Update position
        if p["is_wave"]:
            p["wave_phase"] += 0.2
            base_pos = p["pos"] + p["vel"]
            if p["vel"].length() > 0:
                perp_vec = p["vel"].rotate(90).normalize()
                amplitude = p["radius"] * 1.5
                offset = perp_vec * math.sin(p["wave_phase"]) * amplitude
                p["pos"] = base_pos + offset
            else:
                 p["pos"] = base_pos
        else:
            p["pos"] += p["vel"]

    def _check_collisions(self, p):
        # Particle vs Goal
        if p["pos"].distance_to(self.goal_pos) < p["radius"] + self.goal_radius:
            if not self.game_over:
                self.score = 100
                self.game_over = True
            return []

        # Particle vs Obstacles
        particle_rect = pygame.Rect(p["pos"].x - p["radius"], p["pos"].y - p["radius"], p["radius"] * 2, p["radius"] * 2)
        for obs_rect in self.obstacles:
            if obs_rect.colliderect(particle_rect):
                if p["radius"] < self.MIN_SPLIT_RADIUS:
                    return [] # Particle is too small to split, just disappears

                new_particles = []
                for angle in [-45, 45]:
                    new_vel = p["vel"].rotate(angle) * 1.1
                    new_particle = self._create_particle(
                        pos=p["pos"],
                        vel=new_vel,
                        radius=p["radius"] * 0.75,
                        parent_id=p["id"],
                    )
                    new_particles.append(new_particle)
                return new_particles
        return []

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "timer": self.timer,
            "particle_count": len(self.particles),
        }

    def _render_game(self):
        # Draw Goal
        self._draw_glowing_circle(self.screen, self.COLOR_GOAL, self.goal_pos, self.goal_radius, 15)

        # Draw Obstacles
        for obs_rect in self.obstacles:
            pygame.draw.rect(self.screen, self.COLOR_OBSTACLE, obs_rect)
            pygame.draw.rect(self.screen, tuple(c*0.7 for c in self.COLOR_OBSTACLE), obs_rect, 2)

        # Draw Particles
        for p in self.particles:
            color = self.COLOR_WAVE if p["is_wave"] else p["color"]
            glow_radius = int(p["radius"] * 0.5)
            self._draw_glowing_circle(self.screen, color, p["pos"], int(p["radius"]), glow_radius)

            # Draw velocity vector for main particle
            if p["is_main"] and p["vel"].length() > 0.1:
                end_pos = p["pos"] + p["vel"] * 5
                pygame.draw.line(self.screen, self.COLOR_VECTOR, p["pos"], end_pos, 1)

    def _render_ui(self):
        # Timer
        timer_text = f"TIME: {max(0, self.timer):.1f}"
        timer_surf = self.font_main.render(timer_text, True, self.COLOR_TEXT)
        self.screen.blit(timer_surf, (10, 10))

        # Particle Count
        count_text = f"PARTICLES: {len(self.particles)}"
        count_surf = self.font_main.render(count_text, True, self.COLOR_TEXT)
        self.screen.blit(count_surf, (self.WIDTH - count_surf.get_width() - 10, 10))
        
        # Game Over Message
        if self.game_over:
            message = "GOAL REACHED!" if self.score > 0 else "TIME UP!"
            msg_surf = self.font_main.render(message, True, self.COLOR_TEXT)
            msg_rect = msg_surf.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
            self.screen.blit(msg_surf, msg_rect)

    def _draw_glowing_circle(self, surface, color, center, radius, glow_size):
        if radius <= 0: return
        center_int = (int(center.x), int(center.y))
        
        # Draw multiple layers for glow effect
        for i in range(glow_size, 0, -2):
            alpha = int(150 * (1 - (i / glow_size)))
            if alpha <= 0: continue
            
            # Use gfxdraw for antialiased shapes
            pygame.gfxdraw.filled_circle(
                surface, center_int[0], center_int[1], int(radius + i),
                (color[0], color[1], color[2], alpha)
            )
        
        # Draw the main solid circle on top
        pygame.gfxdraw.aacircle(surface, center_int[0], center_int[1], int(radius), color)
        pygame.gfxdraw.filled_circle(surface, center_int[0], center_int[1], int(radius), color)

    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # This block allows you to play the game manually
    # It will not work with the "dummy" video driver, so we unset it.
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]
        
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((GameEnv.WIDTH, GameEnv.HEIGHT))
    pygame.display.set_caption("Particle Splitter")
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    terminated = False
    
    print("\n--- Manual Control ---")
    print("Arrows: Move | Space: Toggle Wave Form | Q: Quit | Any key to reset after game over")

    while running:
        # --- Pygame Event Handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_q):
                running = False
        
        if terminated:
            # If the episode is over, wait for a key press to reset
            keys = pygame.key.get_pressed()
            if any(keys):
                obs, info = env.reset()
                total_reward = 0
                terminated = False
                continue
        else:
            # --- Action Mapping for Manual Play ---
            keys = pygame.key.get_pressed()
            movement = 0 # No-op
            if keys[pygame.K_UP]: movement = 1
            elif keys[pygame.K_DOWN]: movement = 2
            elif keys[pygame.K_LEFT]: movement = 3
            elif keys[pygame.K_RIGHT]: movement = 4
            
            space_held = 1 if keys[pygame.K_SPACE] else 0
            shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
            
            action = [movement, space_held, shift_held]
            
            # --- Environment Step ---
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward

        # --- Rendering ---
        # The observation is already a rendered frame
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        clock.tick(env.FPS)

    env.close()