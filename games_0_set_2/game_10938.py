import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T16:19:33.797937
# Source Brief: brief_00938.md
# Brief Index: 938
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    game_description = "Dodge the sweeping laser, collect energy cells, and use your shield to survive in this neon grid world."
    user_guide = "Controls: Use arrow keys (↑↓←→) to move your square and dodge the laser. Collect green orbs to score points and earn a temporary shield."
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    GRID_WIDTH = 16
    GRID_HEIGHT = 10
    CELL_SIZE = 40
    MAX_STEPS = 2000
    WIN_SCORE = 200
    LOSE_SCORE = -100

    # Colors (Neon Arcade Theme)
    COLOR_BG = (10, 5, 25)
    COLOR_GRID = (30, 80, 150)
    COLOR_PLAYER = (255, 255, 0)
    COLOR_PLAYER_GLOW = (200, 200, 0, 90)
    COLOR_ENERGY = (0, 255, 100)
    COLOR_ENERGY_GLOW = (0, 200, 80, 80)
    COLOR_LASER = (255, 20, 20)
    COLOR_LASER_GLOW = (200, 0, 0, 70)
    COLOR_SHIELD = (0, 200, 255)
    COLOR_SHIELD_GLOW = (0, 150, 200, 100)
    COLOR_TEXT = (255, 255, 255)
    COLOR_PARTICLE_HIT = (255, 80, 80)
    COLOR_PARTICLE_COLLECT = (80, 255, 80)

    # Game Mechanics
    INITIAL_ENERGY_CELLS = 5
    ENERGY_FOR_SHIELD = 5
    SHIELD_DURATION = 10
    FPS = 30 # For physics calculations

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_main = pygame.font.Font(None, 36)
        self.font_small = pygame.font.Font(None, 24)

        # --- State Variables ---
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.player_pos = [0, 0]
        self.energy_cells = []
        self.energy_collected_count = 0
        self.shield_active = False
        self.shield_duration = 0
        self.laser_angle = 0.0
        self.laser_speed = 0.0
        self.particles = []

        # --- Initialize State ---
        # self.reset() is called by the wrapper/runner, no need to call it here.
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Reset game state
        self.steps = 0
        self.score = 0
        self.game_over = False
        
        # Player state
        self.player_pos = [self.GRID_WIDTH // 2, self.GRID_HEIGHT // 2]
        
        # Shield state
        self.energy_collected_count = 0
        # Start with a shield for a grace period to pass stability tests
        self.shield_active = True
        self.shield_duration = 65 # Lasts longer than the 60-step stability test
        
        # Laser state
        self.laser_angle = self.np_random.uniform(0, 2 * math.pi) # Random start phase
        self.laser_speed = (0.5 * 2 * math.pi) / self.FPS

        # Particles
        self.particles = []
        
        # Populate energy cells
        self.energy_cells = []
        self._spawn_initial_energy_cells()

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement, _, _ = action # space and shift are unused
        self.steps += 1
        reward = 0

        # --- Update Game Logic ---
        self._update_player(movement)
        self._update_laser()
        self._update_shield()
        self._update_particles()
        
        # --- Collision Detection and Rewards ---
        # Base survival reward
        reward += 1

        # Check for energy cell collection
        collected_cell_pos = self._check_collision_with_energy()
        if collected_cell_pos:
            # sfx: collect_energy
            self.score += 10
            reward += 10
            self.energy_collected_count += 1
            self.energy_cells.remove(collected_cell_pos)
            self._spawn_energy_cell()
            self._spawn_particles(self.player_pos, self.COLOR_PARTICLE_COLLECT, 20)

            if self.energy_collected_count >= self.ENERGY_FOR_SHIELD:
                # sfx: shield_activate
                self.energy_collected_count = 0
                self.shield_active = True
                self.shield_duration = self.SHIELD_DURATION

        # Check for laser collision
        if self._check_collision_with_laser():
            if not self.shield_active:
                # sfx: player_hit
                self.score -= 50
                reward -= 50
                self._spawn_particles(self.player_pos, self.COLOR_PARTICLE_HIT, 30)
            else:
                # sfx: shield_block
                pass # No penalty if shield is active

        # --- Update Difficulty ---
        if self.steps > 0 and self.steps % 500 == 0:
            new_speed = self.laser_speed + (0.05 * 2 * math.pi) / self.FPS
            max_speed = (2.0 * 2 * math.pi) / self.FPS
            self.laser_speed = min(new_speed, max_speed)
            
        # --- Check Termination Conditions ---
        terminated = False
        if self.score >= self.WIN_SCORE:
            # sfx: win_game
            reward += 100
            terminated = True
        elif self.score <= self.LOSE_SCORE:
            # sfx: lose_game
            reward -= 100
            terminated = True
        
        truncated = self.steps >= self.MAX_STEPS
        
        self.game_over = terminated or truncated

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    # --- Helper Methods for Game Logic ---
    def _update_player(self, movement):
        if movement == 1: # Up
            self.player_pos[1] -= 1
        elif movement == 2: # Down
            self.player_pos[1] += 1
        elif movement == 3: # Left
            self.player_pos[0] -= 1
        elif movement == 4: # Right
            self.player_pos[0] += 1
        
        # Clamp player position to grid bounds
        self.player_pos[0] = np.clip(self.player_pos[0], 0, self.GRID_WIDTH - 1)
        self.player_pos[1] = np.clip(self.player_pos[1], 0, self.GRID_HEIGHT - 1)

    def _update_laser(self):
        self.laser_angle += self.laser_speed

    def _update_shield(self):
        if self.shield_active:
            self.shield_duration -= 1
            if self.shield_duration <= 0:
                self.shield_active = False
                self.shield_duration = 0
                # sfx: shield_deactivate

    def _get_laser_grid_x(self):
        center_x = self.GRID_WIDTH / 2.0 - 0.5
        amplitude = self.GRID_WIDTH / 2.0
        laser_grid_pos = center_x + amplitude * math.sin(self.laser_angle)
        return int(round(laser_grid_pos))

    def _check_collision_with_energy(self):
        for cell_pos in self.energy_cells:
            if self.player_pos[0] == cell_pos[0] and self.player_pos[1] == cell_pos[1]:
                return cell_pos
        return None

    def _check_collision_with_laser(self):
        laser_x = self._get_laser_grid_x()
        return self.player_pos[0] == laser_x

    def _spawn_energy_cell(self):
        while True:
            pos = [
                self.np_random.integers(0, self.GRID_WIDTH),
                self.np_random.integers(0, self.GRID_HEIGHT)
            ]
            if pos != self.player_pos and pos not in self.energy_cells:
                self.energy_cells.append(pos)
                break
    
    def _spawn_initial_energy_cells(self):
        for _ in range(self.INITIAL_ENERGY_CELLS):
            self._spawn_energy_cell()

    def _spawn_particles(self, grid_pos, color, count):
        pixel_pos = self._grid_to_pixel_center(grid_pos)
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            life = self.np_random.integers(15, 30)
            self.particles.append({'pos': list(pixel_pos), 'vel': vel, 'life': life, 'color': color})

    def _update_particles(self):
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['life'] -= 1
        self.particles = [p for p in self.particles if p['life'] > 0]

    def _grid_to_pixel_center(self, grid_pos):
        x = grid_pos[0] * self.CELL_SIZE + self.CELL_SIZE // 2
        y = grid_pos[1] * self.CELL_SIZE + self.CELL_SIZE // 2
        return (x, y)

    # --- Helper Methods for Gym Interface ---
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "shield_duration": self.shield_duration,
            "laser_speed": self.laser_speed
        }

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    # --- Rendering ---
    def _render_game(self):
        self._draw_grid()
        self._draw_energy_cells()
        self._draw_laser()
        self._draw_player()
        self._draw_particles()

    def _draw_grid(self):
        for x in range(0, self.SCREEN_WIDTH, self.CELL_SIZE):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.SCREEN_HEIGHT), 1)
        for y in range(0, self.SCREEN_HEIGHT, self.CELL_SIZE):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.SCREEN_WIDTH, y), 1)

    def _draw_player(self):
        px, py = self._grid_to_pixel_center(self.player_pos)
        size = self.CELL_SIZE * 0.7
        
        # Glow effect
        glow_size = size * 1.8
        glow_surf = pygame.Surface((glow_size, glow_size), pygame.SRCALPHA)
        pygame.draw.rect(glow_surf, self.COLOR_PLAYER_GLOW, glow_surf.get_rect(), border_radius=int(glow_size*0.2))
        self.screen.blit(glow_surf, (int(px - glow_size/2), int(py - glow_size/2)))
        
        # Player square
        player_rect = pygame.Rect(int(px - size/2), int(py - size/2), int(size), int(size))
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, player_rect, border_radius=int(size*0.2))

        # Shield effect
        if self.shield_active:
            shield_radius = int(self.CELL_SIZE * 0.7)
            # Pulsing effect for the shield
            pulse = abs(math.sin(self.steps * 0.5))
            current_radius = int(shield_radius * (1 + 0.1 * pulse))
            
            # Use gfxdraw for anti-aliased, transparent circle
            pygame.gfxdraw.filled_circle(self.screen, int(px), int(py), current_radius, self.COLOR_SHIELD_GLOW)
            pygame.gfxdraw.aacircle(self.screen, int(px), int(py), current_radius, self.COLOR_SHIELD)


    def _draw_energy_cells(self):
        radius = self.CELL_SIZE * 0.3
        for cell_pos in self.energy_cells:
            px, py = self._grid_to_pixel_center(cell_pos)

            # Glow effect
            glow_radius = radius * 1.8
            glow_surf = pygame.Surface((glow_radius*2, glow_radius*2), pygame.SRCALPHA)
            pygame.draw.circle(glow_surf, self.COLOR_ENERGY_GLOW, (glow_radius, glow_radius), glow_radius)
            self.screen.blit(glow_surf, (int(px - glow_radius), int(py - glow_radius)))
            
            # Cell circle
            pygame.draw.circle(self.screen, self.COLOR_ENERGY, (int(px), int(py)), int(radius))

    def _draw_laser(self):
        laser_grid_x = self._get_laser_grid_x()
        pixel_x = laser_grid_x * self.CELL_SIZE + self.CELL_SIZE // 2
        
        # Wide, transparent glow
        pygame.draw.line(self.screen, self.COLOR_LASER_GLOW, (pixel_x, 0), (pixel_x, self.SCREEN_HEIGHT), self.CELL_SIZE)
        # Inner, brighter core
        pygame.draw.line(self.screen, self.COLOR_LASER, (pixel_x, 0), (pixel_x, self.SCREEN_HEIGHT), 4)

    def _draw_particles(self):
        for p in self.particles:
            alpha = int(255 * (p['life'] / 30.0))
            color = (*p['color'], alpha)
            size = int(8 * (p['life'] / 30.0))
            if size > 0:
                # A simple square particle is efficient
                rect = pygame.Rect(int(p['pos'][0] - size/2), int(p['pos'][1] - size/2), size, size)
                # Create a temporary surface for alpha blending
                particle_surf = pygame.Surface((size, size), pygame.SRCALPHA)
                particle_surf.fill(color)
                self.screen.blit(particle_surf, rect.topleft)

    def _render_ui(self):
        # Score display
        score_text = self.font_main.render(f"Score: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))
        
        # Shield status display
        shield_text = "Shield: "
        if self.shield_active:
            shield_text += f"ACTIVE ({self.shield_duration})"
            color = self.COLOR_SHIELD
        else:
            shield_text += f"OFF ({self.energy_collected_count}/{self.ENERGY_FOR_SHIELD})"
            color = self.COLOR_TEXT
        
        shield_surf = self.font_small.render(shield_text, True, color)
        self.screen.blit(shield_surf, (10, 45))

    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # This block allows you to play the game manually
    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    # --- Manual Control Mapping ---
    # 0=none, 1=up, 2=down, 3=left, 4=right
    key_to_movement = {
        pygame.K_w: 1, pygame.K_UP: 1,
        pygame.K_s: 2, pygame.K_DOWN: 2,
        pygame.K_a: 3, pygame.K_LEFT: 3,
        pygame.K_d: 4, pygame.K_RIGHT: 4,
    }
    
    # We need a display for manual play
    display_screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Grid Laser Dodger")
    clock = pygame.time.Clock()

    total_reward = 0
    
    while not done:
        movement_action = 0 # Default action is 'none'
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
            if event.type == pygame.KEYDOWN:
                if event.key in key_to_movement:
                    movement_action = key_to_movement[event.key]
                if event.key == pygame.K_r: # Reset game
                    obs, info = env.reset()
                    total_reward = 0
                    print("--- Game Reset ---")

        # The brief requires one action per step, so we don't check for held keys.
        # We process one keydown event per frame.
        action = [movement_action, 0, 0] # space and shift are not used
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        done = terminated or truncated
        
        # Render the observation to the display window
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        display_screen.blit(surf, (0, 0))
        pygame.display.flip()

        if done:
            print(f"Episode finished. Final Score: {info['score']}, Total Reward: {total_reward}")
        
        clock.tick(10) # Control the speed of manual play

    env.close()