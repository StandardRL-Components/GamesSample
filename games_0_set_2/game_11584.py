import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T13:11:47.367182
# Source Brief: brief_01584.md
# Brief Index: 1584
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    A Gymnasium environment where the player deploys glowing bacteria
    to illuminate the abyssal trench, terraforming the seabed and
    avoiding horrifying creatures.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Deploy glowing bacteria to illuminate the dark abyssal trench, but beware of the "
        "horrifying creatures that lurk in the shadows."
    )
    user_guide = (
        "Controls: Use arrow keys (↑↓←→) to move the cursor. Press space to deploy a glowing bacteria colony."
    )
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Game Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.MAX_STEPS = 5000
        self.WIN_PERCENTAGE = 95.0

        # Player/Cursor
        self.CURSOR_SPEED = 10.0
        self.PLAYER_RADIUS = 15
        self.PLAYER_POS = np.array([self.WIDTH / 2, self.HEIGHT - 20.0])

        # Bacteria
        self.BACTERIA_MAX_RADIUS = 50
        self.BACTERIA_LIFESPAN = 200
        self.BACTERIA_GROWTH_RATE = 2
        self.BACTERIA_DEPLOY_COOLDOWN = 10
        
        # Creatures
        self.CREATURE_RADIUS = 8
        self.CREATURE_INITIAL_COUNT = 3
        self.CREATURE_INITIAL_SPEED = 1.0
        self.CREATURE_ACCELERATION = 0.001
        self.CREATURE_MAX_SPEED = 5.0
        self.CREATURE_SPAWN_INTERVAL = 500
        self.CREATURE_MAX_COUNT = 20

        # Colors
        self.COLOR_BG_START = (0, 0, 10)
        self.COLOR_BG_END = (5, 0, 30)
        self.COLOR_BACTERIA = (0, 255, 200)
        self.COLOR_CREATURE = (200, 40, 40)
        self.COLOR_PLAYER = (100, 150, 255)
        self.COLOR_CURSOR = (255, 255, 0)
        self.COLOR_UI = (220, 220, 220)

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
        self.font_ui = pygame.font.SysFont("monospace", 20, bold=True)
        
        # --- State Variables ---
        self.steps = 0
        self.score = 0.0
        self.game_over = False
        self.win = False
        self.cursor_pos = np.array([0.0, 0.0])
        self.bacteria = []
        self.creatures = []
        self.deploy_cooldown_timer = 0
        self.milestone_rewards_given = {}
        self.illumination_percentage = 0.0

        # Illumination grid for efficient calculation
        self.grid_res = 20
        self.grid_w = self.WIDTH // self.grid_res
        self.grid_h = self.HEIGHT // self.grid_res
        self.illumination_grid = np.zeros((self.grid_h, self.grid_w), dtype=bool)

        # self.reset() # reset() is called by the environment wrapper, no need to call it here
        # self.validate_implementation() # Validation is useful for dev but not needed in the final class

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0.0
        self.game_over = False
        self.win = False

        self.cursor_pos = np.array([self.WIDTH / 2, self.HEIGHT / 2])
        self.bacteria = []
        self.creatures = []
        self._spawn_initial_creatures()
        
        self.deploy_cooldown_timer = 0
        self.milestone_rewards_given = {'50': False, '75': False}
        self.illumination_grid.fill(False)
        self.illumination_percentage = 0.0

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            terminated = True
            truncated = self.steps >= self.MAX_STEPS
            return self._get_observation(), 0, terminated, truncated, self._get_info()

        self.steps += 1
        reward = 0.0
        prev_illum_percent_int = math.floor(self.illumination_percentage)

        self._handle_input(action)
        self._update_game_logic()
        
        new_illum_percent = self._update_illumination()
        
        # Reward for illumination increase
        if math.floor(new_illum_percent) > prev_illum_percent_int:
            reward += 0.1 * (math.floor(new_illum_percent) - prev_illum_percent_int)

        # Reward for milestones
        if new_illum_percent >= 50 and not self.milestone_rewards_given['50']:
            reward += 5.0
            self.milestone_rewards_given['50'] = True
        if new_illum_percent >= 75 and not self.milestone_rewards_given['75']:
            reward += 10.0
            self.milestone_rewards_given['75'] = True

        # Penalty for nearby creatures
        for creature in self.creatures:
            if np.linalg.norm(creature['pos'] - self.PLAYER_POS) < 100:
                reward -= 0.5
        
        self._check_collisions_and_win_condition()
        
        terminated = self.game_over
        truncated = self.steps >= self.MAX_STEPS

        if terminated:
            if self.win:
                reward += 100.0
            else: # Player consumed
                reward += -100.0
        
        self.score += reward

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _handle_input(self, action):
        movement, space_held, _ = action
        
        # --- Cursor Movement ---
        if movement == 1: self.cursor_pos[1] -= self.CURSOR_SPEED
        elif movement == 2: self.cursor_pos[1] += self.CURSOR_SPEED
        elif movement == 3: self.cursor_pos[0] -= self.CURSOR_SPEED
        elif movement == 4: self.cursor_pos[0] += self.CURSOR_SPEED
        
        self.cursor_pos[0] = np.clip(self.cursor_pos[0], 0, self.WIDTH)
        self.cursor_pos[1] = np.clip(self.cursor_pos[1], 0, self.HEIGHT)

        # --- Bacteria Deployment ---
        if self.deploy_cooldown_timer > 0:
            self.deploy_cooldown_timer -= 1
            
        if space_held and self.deploy_cooldown_timer == 0:
            # sfx: gentle_plop.wav
            self.bacteria.append({
                'pos': self.cursor_pos.copy(),
                'radius': 5,
                'age': 0
            })
            self.deploy_cooldown_timer = self.BACTERIA_DEPLOY_COOLDOWN

    def _update_game_logic(self):
        self._update_bacteria()
        self._update_creatures()
        self._spawn_new_creatures()

    def _update_bacteria(self):
        for b in self.bacteria:
            b['age'] += 1
            if b['radius'] < self.BACTERIA_MAX_RADIUS:
                b['radius'] += self.BACTERIA_GROWTH_RATE
        self.bacteria = [b for b in self.bacteria if b['age'] <= self.BACTERIA_LIFESPAN]

    def _update_creatures(self):
        light_sources = [b['pos'] for b in self.bacteria]
        
        for creature in self.creatures:
            # Increase speed over time
            creature['speed'] = min(self.CREATURE_MAX_SPEED, creature['speed'] + self.CREATURE_ACCELERATION)
            
            if not light_sources:
                continue

            # Find closest light source
            distances = np.linalg.norm(np.array(light_sources) - creature['pos'], axis=1)
            closest_light_idx = np.argmin(distances)
            closest_light_pos = light_sources[closest_light_idx]
            
            # Move away from light
            repulsion_vec = creature['pos'] - closest_light_pos
            dist = np.linalg.norm(repulsion_vec)
            if dist > 0:
                move_vec = (repulsion_vec / dist) * creature['speed']
                creature['pos'] += move_vec
                
            # Clamp to screen
            creature['pos'][0] = np.clip(creature['pos'][0], 0, self.WIDTH)
            creature['pos'][1] = np.clip(creature['pos'][1], 0, self.HEIGHT)

    def _spawn_initial_creatures(self):
        for _ in range(self.CREATURE_INITIAL_COUNT):
            self._spawn_one_creature()

    def _spawn_new_creatures(self):
        if self.steps > 0 and self.steps % self.CREATURE_SPAWN_INTERVAL == 0:
            if len(self.creatures) < self.CREATURE_MAX_COUNT:
                # sfx: deep_rumble.wav
                self._spawn_one_creature()

    def _spawn_one_creature(self):
        # Spawn at the top edge
        pos = np.array([random.uniform(0, self.WIDTH), random.uniform(0, 50)])
        self.creatures.append({'pos': pos, 'speed': self.CREATURE_INITIAL_SPEED})

    def _update_illumination(self):
        self.illumination_grid.fill(False)
        for b in self.bacteria:
            x_min = max(0, int((b['pos'][0] - b['radius']) / self.grid_res))
            x_max = min(self.grid_w, int((b['pos'][0] + b['radius']) / self.grid_res) + 1)
            y_min = max(0, int((b['pos'][1] - b['radius']) / self.grid_res))
            y_max = min(self.grid_h, int((b['pos'][1] + b['radius']) / self.grid_res) + 1)
            
            for y in range(y_min, y_max):
                for x in range(x_min, x_max):
                    grid_center = np.array([(x + 0.5) * self.grid_res, (y + 0.5) * self.grid_res])
                    if np.linalg.norm(grid_center - b['pos']) <= b['radius']:
                        self.illumination_grid[y, x] = True
        
        self.illumination_percentage = np.sum(self.illumination_grid) / self.illumination_grid.size * 100.0
        return self.illumination_percentage

    def _check_collisions_and_win_condition(self):
        # Creature collision with player
        for creature in self.creatures:
            if np.linalg.norm(creature['pos'] - self.PLAYER_POS) < self.PLAYER_RADIUS + self.CREATURE_RADIUS:
                # sfx: creature_attack.wav, player_destroyed.wav
                self.game_over = True
                return
        
        # Win condition
        if self.illumination_percentage >= self.WIN_PERCENTAGE:
            # sfx: victory_chime.wav
            self.game_over = True
            self.win = True

    def _get_observation(self):
        self._render_all()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "illumination": self.illumination_percentage,
            "creatures": len(self.creatures),
        }

    def _render_all(self):
        self._render_background()
        self._render_bacteria()
        self._render_creatures()
        self._render_player()
        self._render_cursor()
        self._render_ui()

    def _render_background(self):
        # A simple gradient for the abyssal depth effect
        for y in range(self.HEIGHT):
            interp = y / self.HEIGHT
            color = (
                self.COLOR_BG_START[0] * (1 - interp) + self.COLOR_BG_END[0] * interp,
                self.COLOR_BG_START[1] * (1 - interp) + self.COLOR_BG_END[1] * interp,
                self.COLOR_BG_START[2] * (1 - interp) + self.COLOR_BG_END[2] * interp,
            )
            pygame.draw.line(self.screen, color, (0, y), (self.WIDTH, y))

    def _render_bacteria(self):
        for b in self.bacteria:
            x, y = int(b['pos'][0]), int(b['pos'][1])
            radius = int(b['radius'])
            life_left = (self.BACTERIA_LIFESPAN - b['age']) / self.BACTERIA_LIFESPAN
            
            # Glow effect with multiple transparent layers
            for i in range(5, 0, -1):
                glow_radius = int(radius * (1 + i * 0.1))
                alpha = int(30 * (1 - (i / 5)) * life_left)
                if glow_radius > 0 and alpha > 0:
                    pygame.gfxdraw.filled_circle(self.screen, x, y, glow_radius, (*self.COLOR_BACTERIA, alpha))
            
            # Core circle
            if radius > 0:
                pygame.gfxdraw.filled_circle(self.screen, x, y, radius, (*self.COLOR_BACTERIA, int(200 * life_left)))
                pygame.gfxdraw.aacircle(self.screen, x, y, radius, (*self.COLOR_BACTERIA, int(255 * life_left)))
    
    def _render_creatures(self):
        for creature in self.creatures:
            pos = creature['pos']
            x, y = int(pos[0]), int(pos[1])
            
            # Simple triangular shape for creature
            p1 = (x, y - self.CREATURE_RADIUS)
            p2 = (x - self.CREATURE_RADIUS // 2, y + self.CREATURE_RADIUS // 2)
            p3 = (x + self.CREATURE_RADIUS // 2, y + self.CREATURE_RADIUS // 2)
            
            pygame.gfxdraw.filled_trigon(self.screen, p1[0], p1[1], p2[0], p2[1], p3[0], p3[1], self.COLOR_CREATURE)
            pygame.gfxdraw.aatrigon(self.screen, p1[0], p1[1], p2[0], p2[1], p3[0], p3[1], self.COLOR_CREATURE)
            # Bioluminescent "eye"
            pygame.gfxdraw.filled_circle(self.screen, x, y, 2, (255, 100, 100))

    def _render_player(self):
        x, y = int(self.PLAYER_POS[0]), int(self.PLAYER_POS[1])
        radius = self.PLAYER_RADIUS
        
        # Glow effect
        for i in range(4, 0, -1):
            glow_radius = int(radius * (1 + i * 0.15))
            alpha = int(40 * (1 - (i / 4)))
            pygame.gfxdraw.filled_circle(self.screen, x, y, glow_radius, (*self.COLOR_PLAYER, alpha))
        
        # Core
        pygame.gfxdraw.filled_circle(self.screen, x, y, radius, self.COLOR_PLAYER)
        pygame.gfxdraw.aacircle(self.screen, x, y, radius, self.COLOR_PLAYER)

    def _render_cursor(self):
        x, y = int(self.cursor_pos[0]), int(self.cursor_pos[1])
        size = 8
        line_width = 2
        
        # Outer glow
        pygame.draw.line(self.screen, (*self.COLOR_CURSOR, 100), (x - size - 4, y), (x + size + 4, y), line_width + 2)
        pygame.draw.line(self.screen, (*self.COLOR_CURSOR, 100), (x, y - size - 4), (x, y + size + 4), line_width + 2)
        
        # Crosshair
        pygame.draw.line(self.screen, self.COLOR_CURSOR, (x - size, y), (x + size, y), line_width)
        pygame.draw.line(self.screen, self.COLOR_CURSOR, (x, y - size), (x, y + size), line_width)

    def _render_ui(self):
        illum_text = f"Illuminated: {self.illumination_percentage:.1f}%"
        score_text = f"Score: {int(self.score)}"
        
        illum_surf = self.font_ui.render(illum_text, True, self.COLOR_UI)
        score_surf = self.font_ui.render(score_text, True, self.COLOR_UI)
        
        self.screen.blit(illum_surf, (10, 10))
        self.screen.blit(score_surf, (self.WIDTH - score_surf.get_width() - 10, 10))
        
        if self.game_over:
            end_text = "VICTORY" if self.win else "TRENCH CONSUMED"
            end_color = self.COLOR_BACTERIA if self.win else self.COLOR_CREATURE
            end_font = pygame.font.SysFont("monospace", 50, bold=True)
            end_surf = end_font.render(end_text, True, end_color)
            end_rect = end_surf.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
            self.screen.blit(end_surf, end_rect)

    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # --- Example Usage ---
    # Un-comment the line below to run with a visible display
    os.environ.setdefault("SDL_VIDEODRIVER", "x11")
    
    env = GameEnv()
    obs, info = env.reset()
    
    running = True
    total_reward = 0
    
    # To run the game with keyboard controls
    manual_screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Abyssal Trench")
    
    while running:
        action = np.array([0, 0, 0]) # [movement, space, shift]
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            
        keys = pygame.key.get_pressed()
        
        if keys[pygame.K_UP]: action[0] = 1
        elif keys[pygame.K_DOWN]: action[0] = 2
        elif keys[pygame.K_LEFT]: action[0] = 3
        elif keys[pygame.K_RIGHT]: action[0] = 4
        
        if keys[pygame.K_SPACE]: action[1] = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: action[2] = 1

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # Render the observation to the manual screen
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        manual_screen.blit(surf, (0, 0))
        pygame.display.flip()

        if terminated or truncated:
            print(f"Episode finished. Final Score: {info['score']:.2f}, Illumination: {info['illumination']:.1f}%")
            # Wait a moment before resetting
            pygame.time.wait(2000)
            obs, info = env.reset()
            total_reward = 0

        env.clock.tick(30) # Limit to 30 FPS for smooth viewing

    env.close()