import gymnasium as gym
import os
import pygame
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame.gfxdraw
import math
import random
from collections import deque
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    """
    A Gymnasium environment where the player controls a cluster of dividing cells.
    The goal is to absorb smaller blue cells to grow, avoid larger red cells,
    and achieve a population of 20 cells within 60 seconds.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Control a colony of cells, absorbing smaller blue cells to grow and multiply. "
        "Avoid the larger red predator cells to survive and reach the target population."
    )
    user_guide = "Controls: Use the arrow keys (↑↓←→) to move your cell colony."
    auto_advance = True

    # --- Constants ---
    WIDTH, HEIGHT = 640, 400
    
    # Colors
    COLOR_BG = (15, 15, 25)
    COLOR_PLAYER = (0, 255, 128)
    COLOR_PLAYER_GLOW = (0, 128, 64)
    COLOR_PREY = (50, 150, 255)
    COLOR_PREY_GLOW = (25, 75, 128)
    COLOR_PREDATOR = (255, 80, 80)
    COLOR_PREDATOR_GLOW = (128, 40, 40)
    COLOR_TEXT = (240, 240, 240)
    COLOR_PARTICLE_ABSORB = (100, 200, 255)
    COLOR_PARTICLE_LOSE = (150, 255, 200)

    # Game Parameters
    TIME_LIMIT_STEPS = 1800  # 60 seconds at 30 FPS
    WIN_CONDITION_CELLS = 20
    PLAYER_SPLIT_TIME = 300 # 10 seconds at 30 FPS

    # Entity Properties
    PLAYER_SPEED = 3.5
    PLAYER_RADIUS = 8
    PREY_SPEED = 1.0
    PREY_RADIUS = 5
    PREDATOR_SPEED = 2.5
    PREDATOR_RADIUS = 14
    
    INITIAL_PREY_COUNT = 15
    INITIAL_PREDATOR_COUNT = 2
    MAX_PREY = 25
    MAX_PREDATORS = 8
    
    # RL Rewards
    REWARD_ABSORB_PREY = 0.5
    REWARD_LOSE_CELL = -1.0
    REWARD_SPLIT = 5.0
    REWARD_WIN = 100.0
    REWARD_LOSE = -100.0

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
        self.font = pygame.font.SysFont("Consolas", 24, bold=True)
        
        # --- Game State Variables (initialized in reset) ---
        self.steps = 0
        self.cumulative_reward = 0.0
        self.game_over = False
        self.win = False
        
        self.player_cells = []
        self.prey_cells = []
        self.predator_cells = []
        self.particles = []
        
        self.split_timer = 0
        self.predator_spawn_timer = 0
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # --- Reset Game State ---
        self.steps = 0
        self.cumulative_reward = 0.0
        self.game_over = False
        self.win = False
        
        # Initialize Player
        self.player_cells = [self._create_cell(self.WIDTH / 2, self.HEIGHT / 2, self.PLAYER_RADIUS)]
        
        # Initialize AI Cells
        self.prey_cells = [self._create_random_cell(self.PREY_RADIUS, self.PREY_SPEED) for _ in range(self.INITIAL_PREY_COUNT)]
        self.predator_cells = [self._create_random_cell(self.PREDATOR_RADIUS, self.PREDATOR_SPEED) for _ in range(self.INITIAL_PREDATOR_COUNT)]
        
        self.particles = []
        self.split_timer = self.PLAYER_SPLIT_TIME
        self.predator_spawn_timer = 90 # 3 seconds at 30 FPS
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self.reset()

        reward = 0.0
        self.steps += 1
        
        # --- Update Game Logic ---
        reward += self._handle_input(action)
        self._update_ai_cells()
        
        collision_rewards = self._handle_collisions()
        reward += collision_rewards
        
        split_reward = self._handle_splitting()
        reward += split_reward
        
        self._handle_spawning()
        self._update_particles()
        
        # --- Check Termination ---
        terminated = False
        truncated = False
        if not self.player_cells:
            terminated = True
            self.game_over = True
            self.win = False
            reward += self.REWARD_LOSE
        elif self.steps >= self.TIME_LIMIT_STEPS:
            terminated = True # Or truncated, depending on definition. Let's use terminated.
            self.game_over = True
            if len(self.player_cells) >= self.WIN_CONDITION_CELLS:
                self.win = True
                reward += self.REWARD_WIN
            else:
                self.win = False
                reward += self.REWARD_LOSE

        self.cumulative_reward += reward
        
        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    # --- Private Helper Methods: Game Logic ---

    def _create_cell(self, x, y, radius, speed=0):
        cell = {
            "pos": pygame.Vector2(x, y),
            "radius": radius,
            "vel": pygame.Vector2(0, 0)
        }
        if speed > 0:
            cell["vel"] = pygame.Vector2(self.np_random.uniform(-1, 1), self.np_random.uniform(-1, 1)).normalize() * speed
            cell["dir_timer"] = self.np_random.integers(60, 120)
        return cell

    def _create_random_cell(self, radius, speed):
        return self._create_cell(
            self.np_random.uniform(radius, self.WIDTH - radius),
            self.np_random.uniform(radius, self.HEIGHT - radius),
            radius,
            speed
        )

    def _handle_input(self, action):
        movement_action = action[0]
        move_vec = pygame.Vector2(0, 0)
        
        if movement_action == 1:  # Up
            move_vec.y = -1
        elif movement_action == 2:  # Down
            move_vec.y = 1
        elif movement_action == 3:  # Left
            move_vec.x = -1
        elif movement_action == 4:  # Right
            move_vec.x = 1
        
        if move_vec.length() > 0:
            move_vec = move_vec.normalize() * self.PLAYER_SPEED

        for cell in self.player_cells:
            cell["pos"] += move_vec
            self._wrap_around(cell)
        
        return 0.0

    def _update_ai_cells(self):
        for cell in self.prey_cells + self.predator_cells:
            cell["pos"] += cell["vel"]
            cell["dir_timer"] -= 1
            if cell["dir_timer"] <= 0:
                cell["vel"] = pygame.Vector2(self.np_random.uniform(-1, 1), self.np_random.uniform(-1, 1)).normalize() * cell["vel"].length()
                cell["dir_timer"] = self.np_random.integers(60, 120)
            self._wrap_around(cell)
            
    def _handle_collisions(self):
        reward = 0.0
        
        # Player vs Prey
        prey_to_remove = []
        for prey in self.prey_cells:
            for player_cell in self.player_cells:
                dist = player_cell["pos"].distance_to(prey["pos"])
                if dist < player_cell["radius"] + prey["radius"]:
                    if prey not in prey_to_remove:
                        prey_to_remove.append(prey)
                        reward += self.REWARD_ABSORB_PREY
                        self._create_particles(prey["pos"], self.COLOR_PARTICLE_ABSORB, 10)
                        new_cell = self._create_cell(player_cell["pos"].x, player_cell["pos"].y, self.PLAYER_RADIUS)
                        self.player_cells.append(new_cell)
                        break 
        self.prey_cells = [p for p in self.prey_cells if p not in prey_to_remove]
        
        # Player vs Predator
        player_cells_to_remove = []
        for player_cell in self.player_cells:
            for predator in self.predator_cells:
                dist = player_cell["pos"].distance_to(predator["pos"])
                if dist < player_cell["radius"] + predator["radius"]:
                    if player_cell not in player_cells_to_remove:
                        player_cells_to_remove.append(player_cell)
                        reward += self.REWARD_LOSE_CELL
                        self._create_particles(player_cell["pos"], self.COLOR_PARTICLE_LOSE, 15)
        
        if player_cells_to_remove:
             self.player_cells = [p for p in self.player_cells if p not in player_cells_to_remove]
             
        return reward

    def _handle_splitting(self):
        self.split_timer -= 1
        if self.split_timer <= 0:
            num_to_split = len(self.player_cells)
            if num_to_split > 0:
                for i in range(num_to_split):
                    parent_cell = self.player_cells[i]
                    new_cell = self._create_cell(parent_cell["pos"].x, parent_cell["pos"].y, self.PLAYER_RADIUS)
                    self.player_cells.append(new_cell)
                    self._create_particles(parent_cell["pos"], self.COLOR_PLAYER, 20, is_flash=True)
                self.split_timer = self.PLAYER_SPLIT_TIME
                return self.REWARD_SPLIT * num_to_split
        return 0.0

    def _handle_spawning(self):
        # Spawn prey
        if len(self.prey_cells) < self.MAX_PREY and self.np_random.random() < 0.1:
            self.prey_cells.append(self._create_random_cell(self.PREY_RADIUS, self.PREY_SPEED))
        
        # Spawn predators
        self.predator_spawn_timer -= 1
        if self.predator_spawn_timer <= 0:
            current_max_predators = min(self.MAX_PREDATORS, self.INITIAL_PREDATOR_COUNT + self.steps // 300) # One more every 10s
            if len(self.predator_cells) < current_max_predators:
                self.predator_cells.append(self._create_random_cell(self.PREDATOR_RADIUS, self.PREDATOR_SPEED))
            self.predator_spawn_timer = self.np_random.integers(60, 180) # 2-6 seconds

    def _wrap_around(self, cell):
        r = cell["radius"]
        if cell["pos"].x < -r: cell["pos"].x = self.WIDTH + r
        if cell["pos"].x > self.WIDTH + r: cell["pos"].x = -r
        if cell["pos"].y < -r: cell["pos"].y = self.HEIGHT + r
        if cell["pos"].y > self.HEIGHT + r: cell["pos"].y = -r
        
    # --- Private Helper Methods: Particles ---
    
    def _create_particles(self, pos, color, count, is_flash=False):
        for _ in range(count):
            if is_flash:
                vel = pygame.Vector2(self.np_random.uniform(-1, 1), self.np_random.uniform(-1, 1)).normalize() * self.np_random.uniform(4, 8)
                life = self.np_random.integers(10, 20)
            else:
                vel = pygame.Vector2(self.np_random.uniform(-1, 1), self.np_random.uniform(-1, 1)).normalize() * self.np_random.uniform(1, 4)
                life = self.np_random.integers(20, 40)
            
            self.particles.append({
                "pos": pos.copy(),
                "vel": vel,
                "life": life,
                "max_life": life,
                "color": color
            })
            
    def _update_particles(self):
        for p in self.particles:
            p["pos"] += p["vel"]
            p["vel"] *= 0.95 # friction
            p["life"] -= 1
        self.particles = [p for p in self.particles if p["life"] > 0]

    # --- Private Helper Methods: Rendering ---

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Render particles
        for p in self.particles:
            alpha = int(255 * (p["life"] / p["max_life"]))
            radius = int(3 * (p["life"] / p["max_life"]))
            if radius > 0:
                pygame.gfxdraw.filled_circle(self.screen, int(p["pos"].x), int(p["pos"].y), radius, (*p["color"], alpha))

        # Render AI cells
        for cell in self.prey_cells:
            self._draw_glowing_circle(self.screen, self.COLOR_PREY, self.COLOR_PREY_GLOW, cell["pos"], cell["radius"])
        for cell in self.predator_cells:
            self._draw_glowing_circle(self.screen, self.COLOR_PREDATOR, self.COLOR_PREDATOR_GLOW, cell["pos"], cell["radius"])
            
        # Render player cells
        for cell in self.player_cells:
            self._draw_glowing_circle(self.screen, self.COLOR_PLAYER, self.COLOR_PLAYER_GLOW, cell["pos"], cell["radius"])

    def _draw_glowing_circle(self, surface, color, glow_color, pos, radius):
        pos_int = (int(pos.x), int(pos.y))
        glow_radius = int(radius * 2.0)
        
        # Draw glow
        s = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
        pygame.draw.circle(s, (*glow_color, 60), (glow_radius, glow_radius), glow_radius)
        surface.blit(s, (pos_int[0] - glow_radius, pos_int[1] - glow_radius), special_flags=pygame.BLEND_RGBA_ADD)

        # Draw main circle
        pygame.gfxdraw.aacircle(surface, pos_int[0], pos_int[1], int(radius), color)
        pygame.gfxdraw.filled_circle(surface, pos_int[0], pos_int[1], int(radius), color)

    def _render_ui(self):
        # Cell count
        cell_text = f"Cells: {len(self.player_cells)}"
        text_surface = self.font.render(cell_text, True, self.COLOR_TEXT)
        self.screen.blit(text_surface, (10, 10))
        
        # Timer
        time_left = max(0, (self.TIME_LIMIT_STEPS - self.steps) / 30)
        time_text = f"Time: {time_left:.1f}"
        text_surface = self.font.render(time_text, True, self.COLOR_TEXT)
        text_rect = text_surface.get_rect(topright=(self.WIDTH - 10, 10))
        self.screen.blit(text_surface, text_rect)
        
        # Game Over / Win message
        if self.game_over:
            msg = "YOU WIN!" if self.win else "GAME OVER"
            color = self.COLOR_PLAYER if self.win else self.COLOR_PREDATOR
            
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 128))
            self.screen.blit(overlay, (0, 0))
            
            end_font = pygame.font.SysFont("Consolas", 72, bold=True)
            text_surface = end_font.render(msg, True, color)
            text_rect = text_surface.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2))
            self.screen.blit(text_surface, text_rect)

    def _get_info(self):
        return {
            "score": self.cumulative_reward,
            "steps": self.steps,
            "player_cells": len(self.player_cells)
        }
    
    def close(self):
        pygame.quit()


# --- Example Usage ---
if __name__ == '__main__':
    env = GameEnv(render_mode="rgb_array")
    
    # --- Pygame window for human play ---
    # Re-initialize pygame for display
    os.environ["SDL_VIDEODRIVER"] = "x11" # Or "windows", "macOS"
    pygame.quit() # Quit the dummy driver
    pygame.init()
    screen = pygame.display.set_mode((GameEnv.WIDTH, GameEnv.HEIGHT))
    pygame.display.set_caption("Cell Growth Environment")
    clock = pygame.time.Clock()

    obs, info = env.reset()
    done = False
    
    # Game loop
    running = True
    while running:
        # --- Human Input ---
        movement = 0 # No-op
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP] or keys[pygame.K_w]: movement = 1
        elif keys[pygame.K_DOWN] or keys[pygame.K_s]: movement = 2
        elif keys[pygame.K_LEFT] or keys[pygame.K_a]: movement = 3
        elif keys[pygame.K_RIGHT] or keys[pygame.K_d]: movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0

        action = [movement, space_held, shift_held]

        # --- Environment Step ---
        obs, reward, terminated, truncated, info = env.step(action)
        
        # --- Rendering ---
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        # --- Event Handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                print("Resetting environment...")
                obs, info = env.reset()
        
        if terminated or truncated:
            print(f"Episode finished. Score: {info['score']}, Cells: {info['player_cells']}, Steps: {info['steps']}")
            # Wait for a moment before auto-resetting
            pygame.time.wait(2000)
            obs, info = env.reset()

        clock.tick(30) # Cap FPS for human play

    env.close()
    pygame.quit()