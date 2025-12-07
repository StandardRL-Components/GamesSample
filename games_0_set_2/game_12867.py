import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T18:34:33.002251
# Source Brief: brief_02867.md
# Brief Index: 2867
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    A Gymnasium environment for a grid-based action-puzzle game.
    The player navigates a grid, firing bubbles to trap enemies.
    The goal is to clear all enemies from the level.
    """
    metadata = {"render_modes": ["rgb_array"]}
    
    game_description = (
        "A grid-based action-puzzle game where you trap moving enemies in bubbles. Clear all enemies to win!"
    )
    user_guide = (
        "Controls: Use arrow keys (↑↓←→) to move on the grid. Press space to fire a bubble."
    )
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.GRID_W, self.GRID_H = 16, 10
        self.CELL_SIZE = 40
        self.FPS = 30
        self.MAX_STEPS = 1500

        # --- Colors ---
        self.COLOR_BG = (15, 15, 35)
        self.COLOR_GRID = (40, 40, 60)
        self.COLOR_TEXT = (240, 240, 255)
        self.COLOR_PLAYER = (50, 255, 150)
        self.COLOR_PLAYER_GLOW = (50, 255, 150, 50)
        self.COLOR_ENEMY_A = (255, 80, 80)
        self.COLOR_ENEMY_B = (80, 150, 255)
        self.COLOR_BUBBLE = (255, 255, 255)
        self.COLOR_BUBBLE_TRAP = (255, 255, 150)
        self.COLOR_POWERUP_SPEED = (255, 220, 0)
        self.COLOR_POWERUP_BIG_BUBBLE = (255, 150, 50)

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
        self.font_ui = pygame.font.SysFont("Consolas", 20, bold=True)
        self.font_powerup = pygame.font.SysFont("Consolas", 16, bold=True)
        self.font_game_over = pygame.font.SysFont("Consolas", 48, bold=True)

        # --- Game State (initialized in reset) ---
        self.level = 1
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False
        self.player_grid_pos = [0, 0]
        self.player_pixel_pos = [0.0, 0.0]
        self.player_target_pixel_pos = [0.0, 0.0]
        self.player_speed = 4.0
        self.bubbles = []
        self.enemies = []
        self.particles = []
        self.powerups = []
        self.active_powerups = {}
        self.enemies_cleared_this_level = 0
        self.prev_space_held = False
        self.np_random = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.level = 1
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False

        self.player_grid_pos = [self.GRID_W // 2, self.GRID_H // 2]
        self.player_pixel_pos = self._grid_to_pixel(self.player_grid_pos)
        self.player_target_pixel_pos = list(self.player_pixel_pos)

        self.bubbles.clear()
        self.enemies.clear()
        self.particles.clear()
        self.powerups.clear()
        self.active_powerups = {'speed': 0, 'big_bubble': 0}
        self.enemies_cleared_this_level = 0
        self.prev_space_held = False

        self._spawn_enemies_for_level()

        return self._get_observation(), self._get_info()

    def step(self, action):
        reward = 0.0
        if self.game_over:
            terminated = True
            truncated = self.steps >= self.MAX_STEPS
            return self._get_observation(), 0, terminated, truncated, self._get_info()

        self.steps += 1
        reward += 0.01  # Survival reward

        self._handle_input(action)
        self._update_player()
        
        trap_reward = self._update_bubbles()
        powerup_reward = self._check_powerup_collection()
        collision_penalty = self._update_enemies()
        
        self._update_active_powerups()
        self._update_particles()

        reward += trap_reward
        reward += powerup_reward
        reward += collision_penalty
        self.score += reward

        terminated = False
        truncated = False

        if collision_penalty < 0:
            self.game_over = True
            terminated = True
            self.score -= 100 # Final penalty
            reward -= 100
        
        if not self.enemies and not self.win:
            self.win = True
            self.game_over = True
            terminated = True
            self.score += 100 # Final reward
            reward += 100
            # SFX: Level Clear
        
        if self.steps >= self.MAX_STEPS:
            truncated = True
            terminated = True # Per Gymnasium API, if truncated is True, terminated should be True

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    # --- Game Logic ---

    def _handle_input(self, action):
        movement, space_held, _ = action[0], action[1] == 1, action[2] == 1

        # Movement is only accepted when the player is at their target cell
        if self.player_pixel_pos == self.player_target_pixel_pos:
            new_grid_pos = list(self.player_grid_pos)
            if movement == 1 and self.player_grid_pos[1] > 0: new_grid_pos[1] -= 1
            elif movement == 2 and self.player_grid_pos[1] < self.GRID_H - 1: new_grid_pos[1] += 1
            elif movement == 3 and self.player_grid_pos[0] > 0: new_grid_pos[0] -= 1
            elif movement == 4 and self.player_grid_pos[0] < self.GRID_W - 1: new_grid_pos[0] += 1
            
            self.player_grid_pos = new_grid_pos
            self.player_target_pixel_pos = self._grid_to_pixel(self.player_grid_pos)

        # Fire bubble on button press (not hold)
        if space_held and not self.prev_space_held and not self.bubbles:
            # SFX: Fire Bubble
            max_radius = 25 + (30 if self.active_powerups.get('big_bubble', 0) > 0 else 0)
            self.bubbles.append({
                'pos': list(self.player_pixel_pos),
                'radius': 5,
                'max_radius': max_radius,
                'timer': 90, # 3 seconds at 30fps
                'state': 'expanding',
                'trapped_enemy': None
            })
        self.prev_space_held = space_held

    def _update_player(self):
        speed = self.player_speed + (4.0 if self.active_powerups.get('speed', 0) > 0 else 0)
        self.player_pixel_pos = self._move_towards(self.player_pixel_pos, self.player_target_pixel_pos, speed)

    def _update_bubbles(self):
        reward = 0
        for bubble in self.bubbles[:]:
            if bubble['state'] == 'expanding':
                bubble['radius'] = min(bubble['max_radius'], bubble['radius'] + 2)
                if bubble['radius'] >= bubble['max_radius']:
                    bubble['state'] = 'floating'
                
                # Check for trapping enemies
                for enemy in self.enemies:
                    if enemy['trapped_in'] is None:
                        dist = math.hypot(bubble['pos'][0] - enemy['pixel_pos'][0], bubble['pos'][1] - enemy['pixel_pos'][1])
                        if dist < bubble['radius']:
                            enemy['trapped_in'] = bubble
                            bubble['trapped_enemy'] = enemy
                            bubble['state'] = 'trapped'
                            reward += 10 # Reward for trapping
                            # SFX: Enemy Trapped
                            break

            elif bubble['state'] == 'floating' or bubble['state'] == 'trapped':
                bubble['timer'] -= 1
                if bubble['timer'] <= 0:
                    bubble['state'] = 'popping'
                    bubble['timer'] = 10 # Pop animation duration
                    # SFX: Bubble Pop
                    if bubble['trapped_enemy']:
                        self.enemies.remove(bubble['trapped_enemy'])
                        self.enemies_cleared_this_level += 1
                        if self.enemies_cleared_this_level % 3 == 0:
                            self._spawn_powerup()
                    self._create_particle_burst(bubble['pos'], self.COLOR_BUBBLE, 30)

            elif bubble['state'] == 'popping':
                bubble['timer'] -= 1
                if bubble['timer'] <= 0:
                    self.bubbles.remove(bubble)
        return reward

    def _update_enemies(self):
        base_speed = 0.5 + (self.level // 5) * 0.25
        for enemy in self.enemies:
            if enemy['trapped_in']:
                # Follow the bubble
                enemy['pixel_pos'] = self._move_towards(enemy['pixel_pos'], enemy['trapped_in']['pos'], 5.0)
                continue

            # Follow path
            speed = base_speed + (1.0 if enemy['type'] == 'B' else 0)
            enemy['pixel_pos'] = self._move_towards(enemy['pixel_pos'], enemy['target_pixel_pos'], speed)
            if enemy['pixel_pos'] == enemy['target_pixel_pos']:
                enemy['path_idx'] = (enemy['path_idx'] + 1) % len(enemy['path'])
                enemy['grid_pos'] = enemy['path'][enemy['path_idx']]
                enemy['target_pixel_pos'] = self._grid_to_pixel(enemy['grid_pos'])

            # Check for player collision
            dist = math.hypot(self.player_pixel_pos[0] - enemy['pixel_pos'][0], self.player_pixel_pos[1] - enemy['pixel_pos'][1])
            if dist < self.CELL_SIZE * 0.4 + self.CELL_SIZE * 0.3:
                # SFX: Player Hit
                self._create_particle_burst(self.player_pixel_pos, self.COLOR_PLAYER, 50)
                return -1.0 # Signal collision
        return 0.0

    def _update_active_powerups(self):
        for key in list(self.active_powerups.keys()):
            if self.active_powerups[key] > 0:
                self.active_powerups[key] -= 1

    def _check_powerup_collection(self):
        reward = 0
        player_rect = pygame.Rect(self.player_pixel_pos[0] - 10, self.player_pixel_pos[1] - 10, 20, 20)
        for powerup in self.powerups[:]:
            powerup_rect = pygame.Rect(powerup['pos'][0] - 10, powerup['pos'][1] - 10, 20, 20)
            if player_rect.colliderect(powerup_rect):
                # SFX: Powerup Collect
                self.active_powerups[powerup['type']] = 300 # 10 seconds at 30fps
                self.powerups.remove(powerup)
                reward += 5.0
        return reward

    def _update_particles(self):
        for p in self.particles[:]:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['vel'][1] += 0.1  # Gravity
            p['life'] -= 1
            if p['life'] <= 0:
                self.particles.remove(p)

    def _spawn_enemies_for_level(self):
        num_enemies = min(2 + self.level, 8)
        for i in range(num_enemies):
            path_type = self.np_random.choice(['h_patrol', 'v_patrol', 'box'])
            start_x = self.np_random.integers(1, self.GRID_W - 2)
            start_y = self.np_random.integers(1, self.GRID_H - 2)
            path = self._generate_path(path_type, [start_x, start_y])
            
            enemy_type = 'B' if self.level >= 10 and i % 3 == 0 else 'A'

            self.enemies.append({
                'type': enemy_type,
                'grid_pos': list(path[0]),
                'pixel_pos': self._grid_to_pixel(path[0]),
                'target_pixel_pos': self._grid_to_pixel(path[1]),
                'path': path,
                'path_idx': 0,
                'trapped_in': None
            })

    def _spawn_powerup(self):
        if len(self.powerups) > 2: return
        pos = [self.np_random.integers(2, self.GRID_W - 2), self.np_random.integers(2, self.GRID_H - 2)]
        ptype = self.np_random.choice(['speed', 'big_bubble'])
        self.powerups.append({'pos': self._grid_to_pixel(pos), 'type': ptype})
        
    def _generate_path(self, ptype, start_pos):
        x, y = start_pos
        w, h = self.GRID_W, self.GRID_H
        if ptype == 'h_patrol':
            return [[x, y], [w - 1 - x, y]]
        if ptype == 'v_patrol':
            return [[x, y], [x, h - 1 - y]]
        if ptype == 'box':
            size = self.np_random.integers(2, 5)
            x1, y1 = x, y
            x2, y2 = min(w - 2, x + size), min(h - 2, y + size)
            return [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
        return [start_pos]

    def _create_particle_burst(self, pos, color, count):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            self.particles.append({
                'pos': list(pos),
                'vel': [math.cos(angle) * speed, math.sin(angle) * speed],
                'life': self.np_random.integers(15, 30),
                'color': color,
                'radius': self.np_random.uniform(1, 4)
            })

    # --- Rendering ---

    def _get_observation(self):
        self._render_all()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_all(self):
        self.screen.fill(self.COLOR_BG)
        self._render_grid()
        for p in self.powerups: self._render_powerup(p)
        for e in self.enemies: self._render_enemy(e)
        self._render_player()
        for b in self.bubbles: self._render_bubble(b)
        for p in self.particles: self._render_particle(p)
        self._render_ui()
        if self.game_over:
            self._render_game_over()

    def _render_grid(self):
        for x in range(0, self.WIDTH, self.CELL_SIZE):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.HEIGHT))
        for y in range(0, self.HEIGHT, self.CELL_SIZE):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.WIDTH, y))

    def _render_player(self):
        pos = (int(self.player_pixel_pos[0]), int(self.player_pixel_pos[1]))
        radius = int(self.CELL_SIZE * 0.35)
        self._draw_glow(self.screen, pos, radius, self.COLOR_PLAYER_GLOW)
        pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], radius, self.COLOR_PLAYER)
        pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], radius, self.COLOR_PLAYER)

    def _render_enemy(self, enemy):
        pos = (int(enemy['pixel_pos'][0]), int(enemy['pixel_pos'][1]))
        size = int(self.CELL_SIZE * 0.6)
        rect = pygame.Rect(pos[0] - size // 2, pos[1] - size // 2, size, size)
        color = self.COLOR_ENEMY_B if enemy['type'] == 'B' else self.COLOR_ENEMY_A
        pygame.draw.rect(self.screen, color, rect, border_radius=3)

    def _render_bubble(self, bubble):
        pos = (int(bubble['pos'][0]), int(bubble['pos'][1]))
        radius = int(bubble['radius'])
        color = self.COLOR_BUBBLE_TRAP if bubble['state'] == 'trapped' else self.COLOR_BUBBLE
        
        if bubble['state'] == 'popping':
            # Visual effect for popping
            alpha = max(0, 255 * (bubble['timer'] / 10))
            temp_surf = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
            pygame.gfxdraw.aacircle(temp_surf, radius, radius, radius, (*color, int(alpha)))
            self.screen.blit(temp_surf, (pos[0] - radius, pos[1] - radius))
        else:
            alpha_color = (*color, 100)
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], radius, alpha_color)
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], radius, color)

    def _render_powerup(self, powerup):
        pos = (int(powerup['pos'][0]), int(powerup['pos'][1]))
        size = int(self.CELL_SIZE * 0.5)
        rect = pygame.Rect(pos[0] - size // 2, pos[1] - size // 2, size, size)
        color = self.COLOR_POWERUP_SPEED if powerup['type'] == 'speed' else self.COLOR_POWERUP_BIG_BUBBLE
        text_char = "S" if powerup['type'] == 'speed' else "B"
        pygame.draw.rect(self.screen, color, rect, border_radius=5)
        text_surf = self.font_powerup.render(text_char, True, self.COLOR_BG)
        text_rect = text_surf.get_rect(center=rect.center)
        self.screen.blit(text_surf, text_rect)

    def _render_particle(self, p):
        alpha = max(0, 255 * (p['life'] / 30))
        color = (*p['color'], int(alpha))
        radius = int(p['radius'])
        temp_surf = pygame.Surface((radius*2, radius*2), pygame.SRCALPHA)
        pygame.draw.circle(temp_surf, color, (radius, radius), radius)
        self.screen.blit(temp_surf, (int(p['pos'][0] - radius), int(p['pos'][1] - radius)))

    def _render_ui(self):
        level_text = self.font_ui.render(f"Level: {self.level}", True, self.COLOR_TEXT)
        self.screen.blit(level_text, (10, 5))
        
        enemies_text = self.font_ui.render(f"Enemies: {len(self.enemies)}", True, self.COLOR_TEXT)
        self.screen.blit(enemies_text, (self.WIDTH - enemies_text.get_width() - 10, 5))

        score_text = self.font_ui.render(f"Score: {int(self.score)}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (self.WIDTH // 2 - score_text.get_width() // 2, 5))

    def _render_game_over(self):
        overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 180))
        self.screen.blit(overlay, (0, 0))
        
        message = "LEVEL CLEARED!" if self.win else "GAME OVER"
        text_surf = self.font_game_over.render(message, True, self.COLOR_TEXT)
        text_rect = text_surf.get_rect(center=(self.WIDTH // 2, self.HEIGHT // 2))
        self.screen.blit(text_surf, text_rect)

    # --- Utility Methods ---

    def _grid_to_pixel(self, grid_pos):
        return [
            grid_pos[0] * self.CELL_SIZE + self.CELL_SIZE // 2,
            grid_pos[1] * self.CELL_SIZE + self.CELL_SIZE // 2
        ]

    def _move_towards(self, current_pos, target_pos, speed):
        dx = target_pos[0] - current_pos[0]
        dy = target_pos[1] - current_pos[1]
        dist = math.hypot(dx, dy)
        if dist <= speed:
            return list(target_pos)
        else:
            return [
                current_pos[0] + (dx / dist) * speed,
                current_pos[1] + (dy / dist) * speed
            ]

    def _draw_glow(self, surface, pos, radius, color):
        temp_surf = pygame.Surface((radius * 4, radius * 4), pygame.SRCALPHA)
        num_layers = 10
        for i in range(num_layers, 0, -1):
            alpha = color[3] * (i / num_layers)**2
            pygame.gfxdraw.filled_circle(
                temp_surf, radius * 2, radius * 2, 
                int(radius * (1 + (num_layers - i) * 0.1)),
                (color[0], color[1], color[2], int(alpha))
            )
        surface.blit(temp_surf, (pos[0] - radius * 2, pos[1] - radius * 2))

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "level": self.level,
            "enemies_remaining": len(self.enemies),
            "win": self.win
        }

    def close(self):
        pygame.quit()

if __name__ == "__main__":
    # Example usage: play the game manually
    # To run with display, comment out the os.environ line at the top of the file
    # or set SDL_VIDEODRIVER to your system's video driver.
    try:
        pygame.display.init()
    except pygame.error:
        print("No video device available, running in headless mode.")
        # Fallback to a dummy main loop if no display
        env = GameEnv()
        obs, info = env.reset(seed=42)
        for _ in range(1000):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                print(f"Episode finished. Final info: {info}")
                obs, info = env.reset(seed=42)
        env.close()
        exit()

    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Bubble Trap")
    clock = pygame.time.Clock()
    
    terminated = False
    action = [0, 0, 0]  # [movement, space, shift]

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    action[1] = 1
            if event.type == pygame.KEYUP:
                if event.key == pygame.K_SPACE:
                    action[1] = 0

        keys = pygame.key.get_pressed()
        action[0] = 0 # No-op
        if keys[pygame.K_UP]: action[0] = 1
        elif keys[pygame.K_DOWN]: action[0] = 2
        elif keys[pygame.K_LEFT]: action[0] = 3
        elif keys[pygame.K_RIGHT]: action[0] = 4
        
        action[2] = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0

        obs, reward, terminated, truncated, info = env.step(action)

        # Pygame uses a different coordinate system, so we need to transpose
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(env.FPS)
        
        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']}, Win: {info['win']}")
            pygame.time.wait(3000) # Pause to show final screen
            obs, info = env.reset()
            terminated = False

    env.close()