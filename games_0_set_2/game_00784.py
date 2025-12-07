
# Generated: 2025-08-27T14:46:09.013014
# Source Brief: brief_00784.md
# Brief Index: 784

        
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

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Use arrow keys to move the cursor. Press Space to build a wall. Survive the waves!"
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Build a procedurally generated block fortress to defend against waves of increasingly difficult enemies in a grid-based strategy game."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((640, 400))
        self.clock = pygame.time.Clock()
        
        # Game constants
        self.GRID_WIDTH, self.GRID_HEIGHT = 32, 20
        self.CELL_SIZE = 20
        self.MAX_WAVES = 20
        self.MAX_STEPS = 1000
        self.INITIAL_FORTRESS_HEALTH = 100
        self.ENEMY_DAMAGE = 10

        # Colors
        self.COLOR_BG = (25, 28, 36)
        self.COLOR_GRID = (45, 48, 56)
        self.COLOR_BLOCK = (66, 179, 139) # Green
        self.COLOR_FORTRESS = (100, 100, 120)
        self.COLOR_FORTRESS_DMG = (255, 80, 80)
        self.COLOR_ENEMY = (224, 85, 85) # Red
        self.COLOR_CURSOR = (255, 255, 255, 100)
        self.COLOR_TEXT = (220, 220, 220)
        self.COLOR_SCORE = (255, 200, 0)
        
        # Fonts
        self.FONT_UI = pygame.font.SysFont("Consolas", 18, bold=True)
        self.FONT_GAMEOVER = pygame.font.SysFont("Consolas", 48, bold=True)

        # Game state variables (initialized in reset)
        self.grid = None
        self.fortress_cells = []
        self.fortress_health = 0
        self.fortress_damage_flash = 0
        self.cursor_pos = [0, 0]
        self.enemies = []
        self.wave = 0
        self.score = 0
        self.steps = 0
        self.game_over = False
        self.rng = None
        
        # Initialize state variables
        self.reset()
        
        # Validate implementation
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.rng = np.random.default_rng(seed)
        
        self._init_grid()
        self.fortress_health = self.INITIAL_FORTRESS_HEALTH
        self.fortress_damage_flash = 0
        self.cursor_pos = [self.GRID_WIDTH // 4, self.GRID_HEIGHT // 2]
        self.wave = 0
        self.score = 0
        self.steps = 0
        self.game_over = False
        self.enemies = []

        self._start_next_wave()
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        reward = 0
        
        movement = action[0]
        space_pressed = action[1] == 1
        
        action_taken = False

        # --- Handle player actions ---
        if movement != 0:
            dx, dy = [(0,0), (0,-1), (0,1), (-1,0), (1,0)][movement]
            self.cursor_pos[0] = max(0, min(self.GRID_WIDTH - 1, self.cursor_pos[0] + dx))
            self.cursor_pos[1] = max(0, min(self.GRID_HEIGHT - 1, self.cursor_pos[1] + dy))
        
        if space_pressed:
            if self._place_block():
                # sfx: BlockPlace
                reward -= 0.1  # Cost for placing a block
                action_taken = True
        
        # --- Update game state if an action occurred ---
        # The game is turn-based; state advances only on block placement.
        if action_taken:
            reward += self._update_enemies()
            wave_reward = self._check_wave_completion()
            reward += wave_reward

        # --- Check for termination ---
        terminated = self._check_termination()
        if terminated and self.fortress_health <= 0:
            reward = -100.0
            # sfx: GameLose
        elif terminated and self.wave > self.MAX_WAVES:
            reward = 100.0
            # sfx: GameWin
        
        self.score += reward

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _init_grid(self):
        self.grid = np.zeros((self.GRID_WIDTH, self.GRID_HEIGHT), dtype=np.int8)
        self.fortress_cells = []
        fortress_y_start = self.GRID_HEIGHT // 2 - 2
        for i in range(4):
            pos = (self.GRID_WIDTH - 2, fortress_y_start + i)
            self.grid[pos] = 2  # 2 = fortress
            self.fortress_cells.append(pos)
    
    def _start_next_wave(self):
        self.wave += 1
        if self.wave > self.MAX_WAVES:
            return
            
        num_enemies = 5 + (self.wave - 1) * 2
        
        spawn_y_positions = self.rng.choice(self.GRID_HEIGHT, size=num_enemies, replace=True)
        for y in spawn_y_positions:
            pos = (0, int(y))
            if self.grid[pos] == 0:
                enemy = {'pos': list(pos), 'path': []}
                self.enemies.append(enemy)
        
        self._recalculate_all_enemy_paths()

    def _place_block(self):
        x, y = self.cursor_pos
        if self.grid[x, y] == 0:
            # Temporarily place block
            self.grid[x, y] = 1
            # Check if this placement traps any enemy or blocks all paths
            if not self._path_exists():
                self.grid[x, y] = 0  # Revert if it blocks all paths
                return False
            self._recalculate_all_enemy_paths()
            return True
        return False

    def _path_exists(self):
        # Check if a path exists from a common spawn area to the fortress
        start_node = (0, self.GRID_HEIGHT // 2)
        if self.grid[start_node] != 0: # If spawn is blocked, try another
            for y_offset in range(1, self.GRID_HEIGHT//2):
                if self.grid[0, (self.GRID_HEIGHT // 2) + y_offset] == 0:
                    start_node = (0, (self.GRID_HEIGHT // 2) + y_offset)
                    break
                if self.grid[0, (self.GRID_HEIGHT // 2) - y_offset] == 0:
                    start_node = (0, (self.GRID_HEIGHT // 2) - y_offset)
                    break
        path = self._bfs(start_node, self.fortress_cells)
        return path is not None

    def _update_enemies(self):
        enemies_to_remove = []
        reward_from_hits = 0
        for enemy in self.enemies:
            if enemy['path'] and len(enemy['path']) > 1:
                enemy['pos'] = list(enemy['path'][1])
                enemy['path'].pop(0)
            
            if tuple(enemy['pos']) in self.fortress_cells:
                self.fortress_health = max(0, self.fortress_health - self.ENEMY_DAMAGE)
                self.fortress_damage_flash = 5  # Visual effect duration
                # sfx: FortressHit
                enemies_to_remove.append(enemy)

        if enemies_to_remove:
            self.enemies = [e for e in self.enemies if e not in enemies_to_remove]
        return reward_from_hits

    def _check_wave_completion(self):
        if not self.enemies and not self.game_over and self.wave <= self.MAX_WAVES:
            # sfx: WaveComplete
            self._start_next_wave()
            return 1.0  # Reward for surviving a wave
        return 0.0

    def _check_termination(self):
        if self.fortress_health <= 0:
            self.game_over = True
        if self.wave > self.MAX_WAVES and not self.enemies:
            self.game_over = True
        if self.steps >= self.MAX_STEPS:
            self.game_over = True
        return self.game_over

    def _recalculate_all_enemy_paths(self):
        for enemy in self.enemies:
            path = self._bfs(tuple(enemy['pos']), self.fortress_cells)
            enemy['path'] = path if path else []

    def _bfs(self, start_node, end_nodes):
        q = [(start_node, [start_node])]
        visited = {start_node}
        
        while q:
            (curr_x, curr_y), path = q.pop(0)
            
            if (curr_x, curr_y) in end_nodes:
                return path

            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                next_x, next_y = curr_x + dx, curr_y + dy
                
                if 0 <= next_x < self.GRID_WIDTH and 0 <= next_y < self.GRID_HEIGHT:
                    if self.grid[next_x, next_y] != 1 and (next_x, next_y) not in visited:
                        visited.add((next_x, next_y))
                        new_path = list(path)
                        new_path.append((next_x, next_y))
                        q.append(((next_x, next_y), new_path))
        return None

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _render_game(self):
        # Draw grid lines
        for x in range(0, self.GRID_WIDTH * self.CELL_SIZE, self.CELL_SIZE):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.GRID_HEIGHT * self.CELL_SIZE))
        for y in range(0, self.GRID_HEIGHT * self.CELL_SIZE, self.CELL_SIZE):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.GRID_WIDTH * self.CELL_SIZE, y))

        # Draw grid elements
        for x in range(self.GRID_WIDTH):
            for y in range(self.GRID_HEIGHT):
                rect = pygame.Rect(x * self.CELL_SIZE, y * self.CELL_SIZE, self.CELL_SIZE, self.CELL_SIZE)
                cell_type = self.grid[x, y]
                if cell_type == 1: # Player block
                    pygame.draw.rect(self.screen, self.COLOR_BLOCK, rect.inflate(-2, -2))
                elif cell_type == 2: # Fortress
                    color = self.COLOR_FORTRESS_DMG if self.fortress_damage_flash > 0 else self.COLOR_FORTRESS
                    pygame.draw.rect(self.screen, color, rect)
        if self.fortress_damage_flash > 0:
            self.fortress_damage_flash -= 1

        # Draw enemies
        for enemy in self.enemies:
            x, y = enemy['pos']
            center = (int((x + 0.5) * self.CELL_SIZE), int((y + 0.5) * self.CELL_SIZE))
            radius = self.CELL_SIZE // 2 - 2
            pygame.gfxdraw.filled_circle(self.screen, center[0], center[1], radius, self.COLOR_ENEMY)
            pygame.gfxdraw.aacircle(self.screen, center[0], center[1], radius, self.COLOR_ENEMY)

        # Draw cursor
        if not self.game_over:
            cursor_rect = pygame.Rect(self.cursor_pos[0] * self.CELL_SIZE, self.cursor_pos[1] * self.CELL_SIZE, self.CELL_SIZE, self.CELL_SIZE)
            cursor_surface = pygame.Surface((self.CELL_SIZE, self.CELL_SIZE), pygame.SRCALPHA)
            cursor_surface.fill(self.COLOR_CURSOR)
            self.screen.blit(cursor_surface, cursor_rect.topleft)

    def _render_ui(self):
        # Health
        health_text = self.FONT_UI.render(f"FORTRESS HP: {self.fortress_health}", True, self.COLOR_TEXT)
        self.screen.blit(health_text, (10, 10))

        # Wave
        wave_text = self.FONT_UI.render(f"WAVE: {self.wave}/{self.MAX_WAVES}", True, self.COLOR_TEXT)
        self.screen.blit(wave_text, (self.screen.get_width() - wave_text.get_width() - 10, 10))
        
        # Score
        score_text = self.FONT_UI.render(f"SCORE: {int(self.score)}", True, self.COLOR_SCORE)
        self.screen.blit(score_text, (self.screen.get_width() - score_text.get_width() - 10, self.screen.get_height() - score_text.get_height() - 10))

        # Game Over message
        if self.game_over:
            overlay = pygame.Surface(self.screen.get_size(), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            msg = "VICTORY!" if self.fortress_health > 0 else "GAME OVER"
            color = self.COLOR_SCORE if self.fortress_health > 0 else self.COLOR_ENEMY
            game_over_text = self.FONT_GAMEOVER.render(msg, True, color)
            text_rect = game_over_text.get_rect(center=self.screen.get_rect().center)
            self.screen.blit(game_over_text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "wave": self.wave,
            "fortress_health": self.fortress_health,
        }

    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation:
        '''
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (400, 640, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (400, 640, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (400, 640, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

# Example of how to run the environment
if __name__ == '__main__':
    env = GameEnv(render_mode="rgb_array")
    
    # --- To display the game in a window ---
    try:
        screen = pygame.display.set_mode((640, 400))
        pygame.display.set_caption("Block Fortress")
        clock = pygame.time.Clock()
        
        obs, info = env.reset()
        done = False
        
        print(env.user_guide)
        
        running = True
        while running:
            # Map pygame keys to gymnasium actions
            keys = pygame.key.get_pressed()
            movement = 0
            if keys[pygame.K_UP]: movement = 1
            elif keys[pygame.K_DOWN]: movement = 2
            elif keys[pygame.K_LEFT]: movement = 3
            elif keys[pygame.K_RIGHT]: movement = 4
            
            space_pressed = 1 if keys[pygame.K_SPACE] else 0
            shift_pressed = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
            
            action = [movement, space_pressed, shift_pressed]

            # In a real game loop, you might only step on an event
            # For this demo, we check for a key press to decide if we step
            action_taken = any(k != 0 for k in action)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_r:
                        obs, info = env.reset()
                        done = False
                    if event.key in [pygame.K_UP, pygame.K_DOWN, pygame.K_LEFT, pygame.K_RIGHT, pygame.K_SPACE]:
                        obs, reward, terminated, truncated, info = env.step(action)
                        done = terminated or truncated
                        print(f"Step: {info['steps']}, Reward: {reward:.2f}, Score: {info['score']:.2f}, Wave: {info['wave']}, Done: {done}")

            # Render the environment's state to the window
            frame = np.transpose(env._get_observation(), (1, 0, 2))
            surf = pygame.surfarray.make_surface(frame)
            screen.blit(surf, (0, 0))
            
            pygame.display.flip()
            clock.tick(30) # Limit frame rate
            
    finally:
        pygame.quit()