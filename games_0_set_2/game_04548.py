
# Generated: 2025-08-28T02:43:51.855270
# Source Brief: brief_04548.md
# Brief Index: 4548

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: Arrow keys to move one grid cell at a time. "
        "Collect yellow gems for points and avoid red enemies."
    )

    game_description = (
        "A fast-paced, grid-based arcade game. Navigate the grid to collect 100 gems while dodging enemies. "
        "The game features three stages of increasing difficulty. Daringly grab bonus gems near enemies for extra points."
    )

    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Game Constants ---
        self.SCREEN_WIDTH, self.SCREEN_HEIGHT = 640, 400
        self.GRID_WIDTH, self.GRID_HEIGHT = 32, 20
        self.CELL_SIZE = 20
        self.MAX_STEPS = 1000
        self.WIN_SCORE = 100
        self.INITIAL_LIVES = 5
        self.STAGE_THRESHOLDS = {1: 25, 2: 60}

        # --- Colors ---
        self.COLOR_BG = (15, 15, 25)
        self.COLOR_GRID = (40, 40, 60)
        self.COLOR_PLAYER = (60, 160, 255)
        self.COLOR_PLAYER_OUTLINE = (200, 220, 255)
        self.COLOR_GEM = (255, 220, 0)
        self.COLOR_GEM_BONUS = (255, 120, 0)
        self.COLOR_ENEMY = (255, 50, 50)
        self.COLOR_ENEMY_OUTLINE = (255, 150, 150)
        self.COLOR_TEXT = (240, 240, 240)
        self.COLOR_HIT_FLASH = (180, 0, 0)

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
        self.font_ui = pygame.font.Font(None, 24)
        self.font_main = pygame.font.Font(None, 48)
        self.font_game_over = pygame.font.Font(None, 72)

        # --- State Variables ---
        self.np_random = None
        self.player_pos = [0, 0]
        self.gems = []
        self.enemies = []
        self.particles = []
        self.score = 0
        self.lives = 0
        self.steps = 0
        self.stage = 1
        self.game_over = False
        self.win_state = False
        self.hit_flash_timer = 0
        self.stage_announce_timer = 0

        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.np_random = np.random.default_rng(seed=seed)
        else:
            self.np_random = np.random.default_rng()

        self.score = 0
        self.lives = self.INITIAL_LIVES
        self.steps = 0
        self.stage = 1
        self.game_over = False
        self.win_state = False
        self.hit_flash_timer = 0
        self.particles = []

        self._setup_stage()
        self.stage_announce_timer = 60 # Show "STAGE 1" for 2 seconds (at 30fps)

        return self._get_observation(), self._get_info()

    def _setup_stage(self):
        self.gems = []
        self.enemies = []
        occupied_cells = set()

        # Place player
        self.player_pos = [self.GRID_WIDTH // 2, self.GRID_HEIGHT - 2]
        occupied_cells.add(tuple(self.player_pos))

        # Place enemies based on stage
        num_enemies = 2 + self.stage * 2
        enemy_speed = 1.0 + (self.stage - 1) * 0.2
        for i in range(num_enemies):
            path_type = self.np_random.choice(['h_patrol', 'v_patrol', 'box_patrol'])
            start_pos = self._get_unoccupied_cell(occupied_cells)
            
            path = []
            if path_type == 'h_patrol':
                length = self.np_random.integers(4, self.GRID_WIDTH // 2)
                for j in range(length): path.append((start_pos[0] + j, start_pos[1]))
                for j in range(length): path.append((start_pos[0] + length - j, start_pos[1]))
            elif path_type == 'v_patrol':
                length = self.np_random.integers(4, self.GRID_HEIGHT // 2)
                for j in range(length): path.append((start_pos[0], start_pos[1] - j))
                for j in range(length): path.append((start_pos[0], start_pos[1] - length + j))
            else: # box_patrol
                size = self.np_random.integers(3, 6)
                for j in range(size): path.append((start_pos[0] + j, start_pos[1]))
                for j in range(size): path.append((start_pos[0] + size, start_pos[1] - j))
                for j in range(size): path.append((start_pos[0] + size - j, start_pos[1] - size))
                for j in range(size): path.append((start_pos[0], start_pos[1] - size + j))
            
            # Filter out-of-bounds path points
            path = [p for p in path if 0 <= p[0] < self.GRID_WIDTH and 0 <= p[1] < self.GRID_HEIGHT]
            if not path: continue

            enemy = {'pos': list(start_pos), 'path': path, 'path_index': 0, 'speed': enemy_speed}
            self.enemies.append(enemy)
            occupied_cells.add(start_pos)

        # Place bonus gems near enemies
        num_bonus_gems = len(self.enemies)
        for enemy in self.enemies:
            for _ in range(5): # Try 5 times to place a bonus gem
                dx, dy = self.np_random.choice([(-1, 0), (1, 0), (0, -1), (0, 1)])
                pos = (enemy['pos'][0] + dx, enemy['pos'][1] + dy)
                if 0 <= pos[0] < self.GRID_WIDTH and 0 <= pos[1] < self.GRID_HEIGHT and pos not in occupied_cells:
                    self.gems.append({'pos': list(pos), 'bonus': True})
                    occupied_cells.add(pos)
                    break
        
        # Place regular gems
        num_gems = 30 - len(self.gems)
        for _ in range(num_gems):
            pos = self._get_unoccupied_cell(occupied_cells)
            self.gems.append({'pos': list(pos), 'bonus': False})
            occupied_cells.add(pos)

    def _get_unoccupied_cell(self, occupied_cells):
        while True:
            pos = (self.np_random.integers(0, self.GRID_WIDTH), self.np_random.integers(0, self.GRID_HEIGHT))
            if pos not in occupied_cells:
                return pos

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        reward = -0.01 # Small penalty for existing, encouraging speed

        # --- Player Movement ---
        movement = action[0]
        prev_pos = list(self.player_pos)
        if movement == 1: self.player_pos[1] -= 1  # Up
        elif movement == 2: self.player_pos[1] += 1  # Down
        elif movement == 3: self.player_pos[0] -= 1  # Left
        elif movement == 4: self.player_pos[0] += 1  # Right
        
        # Clamp player position to grid boundaries
        self.player_pos[0] = np.clip(self.player_pos[0], 0, self.GRID_WIDTH - 1)
        self.player_pos[1] = np.clip(self.player_pos[1], 0, self.GRID_HEIGHT - 1)

        if tuple(self.player_pos) == tuple(prev_pos) and movement != 0:
            reward -= 0.1 # Penalty for bumping into walls

        # --- Enemy Movement ---
        for enemy in self.enemies:
            moves = int(enemy['speed'])
            extra_move_chance = enemy['speed'] - moves
            if self.np_random.random() < extra_move_chance:
                moves += 1
            
            for _ in range(moves):
                if not enemy['path']: continue
                enemy['path_index'] = (enemy['path_index'] + 1) % len(enemy['path'])
                enemy['pos'] = list(enemy['path'][enemy['path_index']])
        
        # --- Collision Detection & State Update ---
        # Gem collision
        gem_to_remove = None
        for gem in self.gems:
            if self.player_pos == gem['pos']:
                if gem['bonus']:
                    self.score += 5
                    reward += 5
                    # sfx: bonus_gem_collect
                else:
                    self.score += 1
                    reward += 1
                    # sfx: gem_collect
                gem_to_remove = gem
                self._spawn_particles(self.player_pos, self.COLOR_GEM if not gem['bonus'] else self.COLOR_GEM_BONUS, 20)
                break
        if gem_to_remove:
            self.gems.remove(gem_to_remove)
            if not self.gems: # Respawn if all collected before stage transition
                self._setup_stage()

        # Enemy collision
        for enemy in self.enemies:
            if self.player_pos == enemy['pos']:
                self.lives -= 1
                reward -= 2 # Penalty for getting hit
                self.hit_flash_timer = 2
                self._spawn_particles(self.player_pos, self.COLOR_ENEMY, 30)
                # sfx: player_hit
                if self.lives <= 0:
                    self.game_over = True
                    reward = -100 # Large terminal penalty for losing
                break
        
        # --- Update timers and particles ---
        if self.hit_flash_timer > 0: self.hit_flash_timer -= 1
        if self.stage_announce_timer > 0: self.stage_announce_timer -= 1
        self._update_particles()

        # --- Stage Progression ---
        if self.stage < 3 and self.score >= self.STAGE_THRESHOLDS[self.stage]:
            self.stage += 1
            reward += 10 # Reward for advancing
            self._setup_stage()
            self.stage_announce_timer = 60
            # sfx: stage_clear

        # --- Termination Conditions ---
        terminated = False
        if self.score >= self.WIN_SCORE:
            self.win_state = True
            self.game_over = True
            reward += 100 # Large terminal reward for winning
            terminated = True
        elif self.lives <= 0:
            self.game_over = True
            terminated = True
        elif self.steps >= self.MAX_STEPS:
            self.game_over = True
            terminated = True

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_grid()
        self._render_gems()
        self._render_enemies()
        self._render_particles()
        self._render_player()
        self._render_ui()
        
        if self.hit_flash_timer > 0:
            flash_surface = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            flash_surface.fill((*self.COLOR_HIT_FLASH, 100))
            self.screen.blit(flash_surface, (0, 0))

        if self.game_over:
            self._render_game_over()
        elif self.stage_announce_timer > 0:
            self._render_stage_announcement()

        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "lives": self.lives,
            "stage": self.stage,
        }

    # --- Rendering Helpers ---
    def _render_grid(self):
        for x in range(0, self.SCREEN_WIDTH, self.CELL_SIZE):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.SCREEN_HEIGHT))
        for y in range(0, self.SCREEN_HEIGHT, self.CELL_SIZE):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.SCREEN_WIDTH, y))

    def _render_player(self):
        x = int((self.player_pos[0] + 0.5) * self.CELL_SIZE)
        y = int((self.player_pos[1] + 0.5) * self.CELL_SIZE)
        radius = int(self.CELL_SIZE * 0.4)
        pygame.gfxdraw.filled_circle(self.screen, x, y, radius, self.COLOR_PLAYER)
        pygame.gfxdraw.aacircle(self.screen, x, y, radius, self.COLOR_PLAYER_OUTLINE)

    def _render_gems(self):
        pulse = (math.sin(self.steps * 0.2) + 1) / 2  # 0 to 1
        for gem in self.gems:
            x = int((gem['pos'][0] + 0.5) * self.CELL_SIZE)
            y = int((gem['pos'][1] + 0.5) * self.CELL_SIZE)
            color = self.COLOR_GEM_BONUS if gem['bonus'] else self.COLOR_GEM
            radius = int(self.CELL_SIZE * (0.25 + pulse * 0.1))
            
            # Draw a diamond shape
            points = [
                (x, y - radius), (x + radius, y),
                (x, y + radius), (x - radius, y)
            ]
            pygame.gfxdraw.aapolygon(self.screen, points, color)
            pygame.gfxdraw.filled_polygon(self.screen, points, color)

    def _render_enemies(self):
        for enemy in self.enemies:
            x = int((enemy['pos'][0] + 0.5) * self.CELL_SIZE)
            y = int((enemy['pos'][1] + 0.5) * self.CELL_SIZE)
            radius = int(self.CELL_SIZE * 0.35)
            pygame.gfxdraw.filled_circle(self.screen, x, y, radius, self.COLOR_ENEMY)
            pygame.gfxdraw.aacircle(self.screen, x, y, radius, self.COLOR_ENEMY_OUTLINE)
            
    def _render_particles(self):
        for p in self.particles:
            alpha = int(255 * (p['life'] / p['max_life']))
            color = (*p['color'], alpha)
            pos = (int(p['pos'][0]), int(p['pos'][1]))
            pygame.draw.circle(self.screen, color, pos, int(p['size']))

    def _render_ui(self):
        # Score
        score_text = self.font_ui.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))

        # Stage
        stage_text = self.font_ui.render(f"STAGE: {self.stage}", True, self.COLOR_TEXT)
        self.screen.blit(stage_text, (self.SCREEN_WIDTH - stage_text.get_width() - 10, 10))
        
        # Lives
        lives_text = self.font_ui.render("LIVES:", True, self.COLOR_TEXT)
        self.screen.blit(lives_text, (self.SCREEN_WIDTH // 2 - 50, 10))
        for i in range(self.lives):
            heart_pos_x = self.SCREEN_WIDTH // 2 + 10 + (i * 20)
            heart_pos_y = 18
            pygame.gfxdraw.filled_circle(self.screen, heart_pos_x, heart_pos_y, 6, self.COLOR_ENEMY)
            pygame.gfxdraw.aacircle(self.screen, heart_pos_x, heart_pos_y, 6, self.COLOR_ENEMY)


    def _render_game_over(self):
        overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 180))
        self.screen.blit(overlay, (0, 0))
        
        text = "YOU WIN!" if self.win_state else "GAME OVER"
        color = self.COLOR_GEM if self.win_state else self.COLOR_ENEMY
        
        title_surf = self.font_game_over.render(text, True, color)
        title_rect = title_surf.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2 - 20))
        self.screen.blit(title_surf, title_rect)

        score_surf = self.font_main.render(f"Final Score: {self.score}", True, self.COLOR_TEXT)
        score_rect = score_surf.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2 + 40))
        self.screen.blit(score_surf, score_rect)

    def _render_stage_announcement(self):
        alpha = min(255, int(255 * (self.stage_announce_timer / 30.0)))
        text_surf = self.font_main.render(f"STAGE {self.stage}", True, self.COLOR_TEXT)
        text_surf.set_alpha(alpha)
        text_rect = text_surf.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
        self.screen.blit(text_surf, text_rect)

    # --- Logic Helpers ---
    def _spawn_particles(self, grid_pos, color, count):
        x = (grid_pos[0] + 0.5) * self.CELL_SIZE
        y = (grid_pos[1] + 0.5) * self.CELL_SIZE
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            self.particles.append({
                'pos': [x, y],
                'vel': [math.cos(angle) * speed, math.sin(angle) * speed],
                'life': self.np_random.integers(10, 20),
                'max_life': 20,
                'size': self.np_random.uniform(1, 4),
                'color': color
            })
    
    def _update_particles(self):
        self.particles = [p for p in self.particles if p['life'] > 0]
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['life'] -= 1
            p['vel'][1] += 0.1 # Gravity

    def validate_implementation(self):
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game directly
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    pygame.display.set_caption("Gem Grid")
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    while running:
        action = np.array([0, 0, 0]) # Default action: no-op
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]:
            action[0] = 1
        elif keys[pygame.K_DOWN]:
            action[0] = 2
        elif keys[pygame.K_LEFT]:
            action[0] = 3
        elif keys[pygame.K_RIGHT]:
            action[0] = 4
        
        # In this turn-based game, we only step when a key is pressed.
        # To make it playable, we'll check if any movement key was pressed.
        if action[0] != 0:
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            
            print(f"Step: {info['steps']}, Score: {info['score']}, Lives: {info['lives']}, Reward: {reward:.2f}")

            if terminated or truncated:
                print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
                obs, info = env.reset()
                total_reward = 0
                pygame.time.wait(2000) # Pause before restarting

        # Render the observation from the environment
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        clock.tick(10) # Limit player input speed

    pygame.quit()