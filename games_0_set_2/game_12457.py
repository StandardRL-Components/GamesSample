import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T14:40:04.758487
# Source Brief: brief_02457.md
# Brief Index: 2457
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
from collections import deque

class GameEnv(gym.Env):
    """
    A Gymnasium environment where the player controls matter particles to annihilate
    incoming antimatter waves in a visually rich particle simulation.
    """
    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}

    game_description = (
        "Control a squadron of matter particles to intercept and annihilate incoming waves of antimatter. "
        "Strategically move your particles to clear all waves and achieve a high score."
    )
    user_guide = (
        "Use the arrow keys (↑↓←→) to move the selected matter particle. "
        "Press Space to cycle to the next particle and Shift to cycle to the previous one."
    )
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.MAX_STEPS = 1000
        self.TOTAL_WAVES = 10
        self.INITIAL_MATTER_COUNT = 5

        # Colors (Primary interactive elements are bright, background is dark)
        self.COLOR_BG = (10, 15, 25)
        self.COLOR_GRID = (30, 40, 60)
        self.COLOR_MATTER = (0, 150, 255)
        self.COLOR_MATTER_GLOW = (100, 200, 255)
        self.COLOR_ANTIMATTER = (255, 50, 50)
        self.COLOR_ANTIMATTER_GLOW = (255, 150, 150)
        self.COLOR_EXPLOSION = (255, 255, 255)
        self.COLOR_UI_TEXT = (220, 220, 240)

        # Physics & Gameplay
        self.PLAYER_SPEED = 4.0
        self.PLAYER_DRAG = 0.90
        self.PARTICLE_RADIUS = 8
        self.TRAIL_LENGTH = 15
        self.EXPLOSION_SPEED = 2.0
        self.EXPLOSION_MAX_RADIUS = 50

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
        try:
            self.font_ui = pygame.font.SysFont("Consolas", 20, bold=True)
            self.font_score = pygame.font.SysFont("Consolas", 28, bold=True)
        except pygame.error:
            self.font_ui = pygame.font.Font(None, 24)
            self.font_score = pygame.font.Font(None, 32)

        # --- State Variables ---
        self.steps = 0
        self.score = 0
        self.wave_number = 0
        self.game_over = False
        self.matter_particles = []
        self.antimatter_particles = []
        self.explosions = []
        self.selected_matter_index = 0
        self.space_was_held = False
        self.shift_was_held = False


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.wave_number = 1
        self.game_over = False
        self.explosions = []
        self.selected_matter_index = 0
        self.space_was_held = False
        self.shift_was_held = False

        # Create initial matter particles
        self.matter_particles = []
        for i in range(self.INITIAL_MATTER_COUNT):
            angle = (2 * math.pi / self.INITIAL_MATTER_COUNT) * i
            x = self.WIDTH / 2 + math.cos(angle) * self.WIDTH / 4
            y = self.HEIGHT / 2 + math.sin(angle) * self.HEIGHT / 4
            self.matter_particles.append({
                "pos": np.array([x, y], dtype=float),
                "vel": np.array([0.0, 0.0], dtype=float),
                "trail": deque(maxlen=self.TRAIL_LENGTH)
            })

        self._start_new_wave()

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            # If the game has ended, do nothing but return the final state
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        reward = 0

        # --- 1. Process Actions ---
        self._handle_input(action)

        # --- 2. Update Game State ---
        self._update_particles()
        reward_from_collisions = self._handle_collisions()
        reward += reward_from_collisions
        self._update_explosions()

        # --- 3. Check Game Flow (Wave progression, Win/Loss) ---
        terminated = False
        truncated = False
        if not self.antimatter_particles and not self.game_over:
            # Wave cleared
            reward += 1.0
            self.wave_number += 1
            if self.wave_number > self.TOTAL_WAVES:
                # Victory condition
                self.game_over = True
                terminated = True
                reward += 100.0
            else:
                self._start_new_wave()

        if not self.matter_particles and not self.game_over:
            # Loss condition
            self.game_over = True
            terminated = True
            reward -= 100.0

        if self.steps >= self.MAX_STEPS:
            truncated = True
        
        terminated = self.game_over or terminated

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _handle_input(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1

        # Handle particle selection (on button press, not hold)
        if self.matter_particles:
            if space_held and not self.space_was_held:
                self.selected_matter_index = (self.selected_matter_index + 1) % len(self.matter_particles)
                # Sound effect placeholder: # sfx_select_up()
            if shift_held and not self.shift_was_held:
                self.selected_matter_index = (self.selected_matter_index - 1 + len(self.matter_particles)) % len(self.matter_particles)
                # Sound effect placeholder: # sfx_select_down()

        self.space_was_held = space_held
        self.shift_was_held = shift_held

        # Handle movement
        if self.matter_particles:
            selected_particle = self.matter_particles[self.selected_matter_index]
            if movement == 1: # Up
                selected_particle["vel"][1] -= self.PLAYER_SPEED
            elif movement == 2: # Down
                selected_particle["vel"][1] += self.PLAYER_SPEED
            elif movement == 3: # Left
                selected_particle["vel"][0] -= self.PLAYER_SPEED
            elif movement == 4: # Right
                selected_particle["vel"][0] += self.PLAYER_SPEED


    def _update_particles(self):
        # Update matter particles
        for p in self.matter_particles:
            p["vel"] *= self.PLAYER_DRAG
            p["pos"] += p["vel"] / self.metadata["render_fps"]
            p["trail"].append(tuple(p["pos"]))
            # Screen wrap-around
            p["pos"][0] %= self.WIDTH
            p["pos"][1] %= self.HEIGHT

        # Update antimatter particles
        center = np.array([self.WIDTH / 2, self.HEIGHT / 2])
        speed = 0.5 + (self.wave_number - 1) * 0.05
        for p in self.antimatter_particles:
            direction = center - p["pos"]
            dist = np.linalg.norm(direction)
            if dist > 1:
                p["vel"] = (direction / dist) * speed
            else:
                p["vel"] = np.array([0.0, 0.0])
            p["pos"] += p["vel"]
            p["trail"].append(tuple(p["pos"]))
            # Screen wrap-around
            p["pos"][0] %= self.WIDTH
            p["pos"][1] %= self.HEIGHT

    def _handle_collisions(self):
        reward_this_step = 0
        matter_to_remove = set()
        antimatter_to_remove = set()

        for i, matter in enumerate(self.matter_particles):
            for j, antimatter in enumerate(self.antimatter_particles):
                if j in antimatter_to_remove:
                    continue
                dist = np.linalg.norm(matter["pos"] - antimatter["pos"])
                if dist < self.PARTICLE_RADIUS * 2:
                    matter_to_remove.add(i)
                    antimatter_to_remove.add(j)
                    reward_this_step += 0.1
                    self.score += 10
                    self.explosions.append({
                        "pos": tuple(matter["pos"]),
                        "radius": self.PARTICLE_RADIUS,
                        "alpha": 255
                    })
                    # Sound effect placeholder: # sfx_annihilate()
                    break # A matter particle can only annihilate one thing per frame

        if matter_to_remove or antimatter_to_remove:
            # Rebuild lists, excluding removed particles
            original_matter_count = len(self.matter_particles)
            old_selected_particle_pos = self.matter_particles[self.selected_matter_index]["pos"] if self.matter_particles else None

            self.matter_particles = [p for i, p in enumerate(self.matter_particles) if i not in matter_to_remove]
            self.antimatter_particles = [p for j, p in enumerate(self.antimatter_particles) if j not in antimatter_to_remove]

            # Adjust selection index to be stable
            if self.matter_particles:
                if len(self.matter_particles) < original_matter_count and old_selected_particle_pos is not None:
                    # Find the particle closest to the previously selected one
                    closest_dist = float('inf')
                    new_idx = 0
                    for i, p in enumerate(self.matter_particles):
                        dist = np.linalg.norm(p["pos"] - old_selected_particle_pos)
                        if dist < closest_dist:
                            closest_dist = dist
                            new_idx = i
                    self.selected_matter_index = new_idx
                self.selected_matter_index = min(self.selected_matter_index, len(self.matter_particles) - 1)
            else:
                self.selected_matter_index = 0

        return reward_this_step

    def _update_explosions(self):
        for e in self.explosions:
            e["radius"] += self.EXPLOSION_SPEED
            e["alpha"] = max(0, e["alpha"] - (255 / (self.EXPLOSION_MAX_RADIUS / self.EXPLOSION_SPEED)))
        self.explosions = [e for e in self.explosions if e["alpha"] > 0]

    def _start_new_wave(self):
        self.antimatter_particles = []
        num_antimatter = 5 + (self.wave_number - 1) * 2
        for _ in range(num_antimatter):
            # Spawn at edges
            edge = self.np_random.integers(4)
            if edge == 0: # top
                x, y = self.np_random.uniform(0, self.WIDTH), -self.PARTICLE_RADIUS
            elif edge == 1: # bottom
                x, y = self.np_random.uniform(0, self.WIDTH), self.HEIGHT + self.PARTICLE_RADIUS
            elif edge == 2: # left
                x, y = -self.PARTICLE_RADIUS, self.np_random.uniform(0, self.HEIGHT)
            else: # right
                x, y = self.WIDTH + self.PARTICLE_RADIUS, self.np_random.uniform(0, self.HEIGHT)
            self.antimatter_particles.append({
                "pos": np.array([x, y], dtype=float),
                "vel": np.array([0.0, 0.0], dtype=float),
                "trail": deque(maxlen=self.TRAIL_LENGTH)
            })
        # Sound effect placeholder: # sfx_new_wave()

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_grid()
        self._render_trails()
        self._render_explosions()
        self._render_particles()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "wave": self.wave_number,
            "matter_remaining": len(self.matter_particles)
        }

    def _render_grid(self):
        for x in range(0, self.WIDTH, 40):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.HEIGHT))
        for y in range(0, self.HEIGHT, 40):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.WIDTH, y))

    def _render_trails(self):
        for p_list, color in [(self.matter_particles, self.COLOR_MATTER), (self.antimatter_particles, self.COLOR_ANTIMATTER)]:
            for p in p_list:
                if len(p["trail"]) > 1:
                    points = [tuple(map(int, pos)) for pos in p["trail"]]
                    for i in range(len(points) - 1):
                        alpha = int(255 * (i / self.TRAIL_LENGTH))
                        trail_color = (*color, alpha)
                        try:
                            pygame.draw.line(self.screen, trail_color, points[i], points[i+1], 1)
                        except TypeError: # Color might have alpha, which line does not support directly
                             temp_surf = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
                             pygame.draw.line(temp_surf, trail_color, points[i], points[i+1], 1)
                             self.screen.blit(temp_surf, (0,0))


    def _render_particles(self):
        # Antimatter
        for p in self.antimatter_particles:
            self._draw_glowing_circle(p["pos"], self.COLOR_ANTIMATTER, self.COLOR_ANTIMATTER_GLOW, self.PARTICLE_RADIUS)

        # Matter
        for i, p in enumerate(self.matter_particles):
            is_selected = (i == self.selected_matter_index)
            radius = self.PARTICLE_RADIUS + 2 if is_selected else self.PARTICLE_RADIUS
            self._draw_glowing_circle(p["pos"], self.COLOR_MATTER, self.COLOR_MATTER_GLOW, radius)
            if is_selected: # Draw selection indicator
                angle = (pygame.time.get_ticks() / 200) % (2 * math.pi)
                indicator_pos = (
                    int(p["pos"][0] + math.cos(angle) * (radius + 5)),
                    int(p["pos"][1] + math.sin(angle) * (radius + 5))
                )
                pygame.gfxdraw.aacircle(self.screen, indicator_pos[0], indicator_pos[1], 2, self.COLOR_UI_TEXT)

    def _draw_glowing_circle(self, pos, color, glow_color, radius):
        x, y = int(pos[0]), int(pos[1])
        # Glow effect
        pygame.gfxdraw.filled_circle(self.screen, x, y, radius + 3, (*glow_color, 50))
        pygame.gfxdraw.filled_circle(self.screen, x, y, radius + 1, (*glow_color, 100))
        # Main particle
        pygame.gfxdraw.aacircle(self.screen, x, y, radius, color)
        pygame.gfxdraw.filled_circle(self.screen, x, y, radius, color)

    def _render_explosions(self):
        for e in self.explosions:
            x, y = int(e["pos"][0]), int(e["pos"][1])
            radius = int(e["radius"])
            alpha = int(e["alpha"])
            color = (*self.COLOR_EXPLOSION, alpha)
            if radius > 0:
                pygame.gfxdraw.aacircle(self.screen, x, y, radius, color)

    def _render_ui(self):
        # Wave number
        wave_text = self.font_ui.render(f"WAVE: {self.wave_number}/{self.TOTAL_WAVES}", True, self.COLOR_UI_TEXT)
        self.screen.blit(wave_text, (10, 10))

        # Matter particles remaining
        matter_text = self.font_ui.render(f"MATTER: {len(self.matter_particles)}", True, self.COLOR_UI_TEXT)
        self.screen.blit(matter_text, (self.WIDTH - matter_text.get_width() - 10, 10))

        # Score
        score_text = self.font_score.render(f"{self.score:06d}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (self.WIDTH / 2 - score_text.get_width() / 2, self.HEIGHT - score_text.get_height() - 5))

    def close(self):
        pygame.font.quit()
        pygame.quit()

if __name__ == '__main__':
    # This block allows you to play the game manually for testing
    env = GameEnv()
    obs, info = env.reset()
    
    # Re-initialize pygame with a display for manual play
    os.environ.pop("SDL_VIDEODRIVER", None)
    pygame.quit()
    pygame.init()
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Particle Annihilator")
    
    terminated = False
    truncated = False
    clock = pygame.time.Clock()
    
    while not terminated and not truncated:
        # --- Human Input ---
        movement = 0 # No-op
        space_held = 0
        shift_held = 0
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP] or keys[pygame.K_w]: movement = 1
        elif keys[pygame.K_DOWN] or keys[pygame.K_s]: movement = 2
        elif keys[pygame.K_LEFT] or keys[pygame.K_a]: movement = 3
        elif keys[pygame.K_RIGHT] or keys[pygame.K_d]: movement = 4
        
        if keys[pygame.K_SPACE]: space_held = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift_held = 1
        
        action = [movement, space_held, shift_held]
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()

        # --- Gym Step ---
        obs, reward, term, trunc, info = env.step(action)
        terminated = term
        truncated = trunc
        
        if reward != 0:
            print(f"Step: {info['steps']}, Score: {info['score']}, Reward: {reward:.2f}")

        # --- Render ---
        # The observation is the rendered frame, so we just need to display it
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(env.metadata["render_fps"])

    env.close()
    print(f"Game Over! Final Score: {info['score']}")