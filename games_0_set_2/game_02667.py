
# Generated: 2025-08-28T05:33:15.769604
# Source Brief: brief_02667.md
# Brief Index: 2667

        
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
        "Controls: Arrow keys to move your blue square. Collect yellow gems and avoid red enemies."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Navigate a grid, collecting gems while dodging enemies. Collect 20 gems to win, but 3 hits and you're out!"
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False

    # --- Constants ---
    GRID_WIDTH = 32
    GRID_HEIGHT = 20
    CELL_SIZE = 20
    SCREEN_WIDTH = GRID_WIDTH * CELL_SIZE
    SCREEN_HEIGHT = GRID_HEIGHT * CELL_SIZE

    MAX_STEPS = 1000
    WIN_CONDITION_GEMS = 20
    MAX_LIVES = 3
    INITIAL_ENEMIES = 5
    INITIAL_GEMS = 5

    # Colors
    COLOR_BG = (15, 15, 25) # Dark blue/black
    COLOR_GRID = (40, 40, 60)
    COLOR_PLAYER = (50, 150, 255)
    COLOR_GEM = (255, 220, 50)
    COLOR_ENEMY = (255, 50, 50)
    COLOR_TEXT = (240, 240, 240)
    COLOR_HEART = (255, 80, 80)
    COLOR_HEART_EMPTY = (70, 70, 90)

    # Rewards
    REWARD_GEM_COLLECT = 10
    REWARD_WIN = 100
    PENALTY_ENEMY_HIT = -5
    PENALTY_LOSS = -100
    REWARD_CLOSER_TO_GEM = 1
    PENALTY_CLOSER_TO_ENEMY = -1

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.Font(None, 24)
        self.font_large = pygame.font.Font(None, 48)
        
        # Game state variables are initialized in reset()
        self.player_pos = None
        self.enemies = None
        self.gem_positions = None
        self.steps = None
        self.score = None
        self.lives = None
        self.gems_collected = None
        self.game_over = None
        self.np_random = None

        self.enemy_move_pattern = [(0, -1), (1, 0), (0, 1), (-1, 0)] # Up, Right, Down, Left

        # self.validate_implementation() # Call for development verification

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.np_random = np.random.default_rng(seed)
        
        # Initialize all game state
        self.steps = 0
        self.score = 0
        self.lives = self.MAX_LIVES
        self.gems_collected = 0
        self.game_over = False

        occupied_positions = set()

        # Place player
        self.player_pos = (self.GRID_WIDTH // 2, self.GRID_HEIGHT // 2)
        occupied_positions.add(self.player_pos)

        # Place enemies
        self.enemies = []
        for _ in range(self.INITIAL_ENEMIES):
            pos = self._get_empty_location(occupied_positions)
            self.enemies.append({
                'pos': pos,
                'pattern_index': self.np_random.integers(0, len(self.enemy_move_pattern))
            })
            occupied_positions.add(pos)

        # Place gems
        self.gem_positions = []
        for _ in range(self.INITIAL_GEMS):
            pos = self._get_empty_location(occupied_positions)
            self.gem_positions.append(pos)
            occupied_positions.add(pos)
        
        return self._get_observation(), self._get_info()

    def _get_empty_location(self, occupied):
        while True:
            pos = (
                self.np_random.integers(0, self.GRID_WIDTH),
                self.np_random.integers(0, self.GRID_HEIGHT)
            )
            if pos not in occupied:
                return pos

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement = action[0]
        reward = 0
        terminated = False

        # --- 1. Calculate pre-move distances for reward shaping ---
        old_dist_gem = self._get_min_dist_to_target(self.player_pos, self.gem_positions)
        old_dist_enemy = self._get_min_dist_to_target(self.player_pos, [e['pos'] for e in self.enemies])
        
        # --- 2. Update player position ---
        if movement != 0: # 0 is no-op
            dx, dy = 0, 0
            if movement == 1: dy = -1 # Up
            elif movement == 2: dy = 1  # Down
            elif movement == 3: dx = -1 # Left
            elif movement == 4: dx = 1  # Right
            
            self.player_pos = (
                (self.player_pos[0] + dx) % self.GRID_WIDTH,
                (self.player_pos[1] + dy) % self.GRID_HEIGHT
            )
        
        # --- 3. Update enemy positions ---
        for enemy in self.enemies:
            move = self.enemy_move_pattern[enemy['pattern_index']]
            enemy['pos'] = (
                (enemy['pos'][0] + move[0]) % self.GRID_WIDTH,
                (enemy['pos'][1] + move[1]) % self.GRID_HEIGHT
            )
            enemy['pattern_index'] = (enemy['pattern_index'] + 1) % len(self.enemy_move_pattern)

        # --- 4. Handle collisions and events ---
        # Gem collection
        if self.player_pos in self.gem_positions:
            # sfx: gem collect sound
            self.gem_positions.remove(self.player_pos)
            self.score += self.REWARD_GEM_COLLECT
            reward += self.REWARD_GEM_COLLECT
            self.gems_collected += 1
            
            # Spawn new gem
            occupied = {self.player_pos} | {e['pos'] for e in self.enemies} | set(self.gem_positions)
            new_gem_pos = self._get_empty_location(occupied)
            self.gem_positions.append(new_gem_pos)

        # Enemy collision
        enemy_positions = [e['pos'] for e in self.enemies]
        if self.player_pos in enemy_positions:
            # sfx: player hit sound
            self.lives -= 1
            self.score += self.PENALTY_ENEMY_HIT
            reward += self.PENALTY_ENEMY_HIT
            # Reset player to a safe, empty location to avoid multi-hits
            occupied = {e['pos'] for e in self.enemies} | set(self.gem_positions)
            self.player_pos = self._get_empty_location(occupied)


        # --- 5. Calculate post-move distance rewards ---
        new_dist_gem = self._get_min_dist_to_target(self.player_pos, self.gem_positions)
        new_dist_enemy = self._get_min_dist_to_target(self.player_pos, [e['pos'] for e in self.enemies])

        if new_dist_gem < old_dist_gem:
            reward += self.REWARD_CLOSER_TO_GEM
        if new_dist_enemy < old_dist_enemy:
            reward += self.PENALTY_CLOSER_TO_ENEMY

        # --- 6. Update step counter and check for termination ---
        self.steps += 1
        
        if self.gems_collected >= self.WIN_CONDITION_GEMS:
            # sfx: win fanfare
            reward += self.REWARD_WIN
            self.score += self.REWARD_WIN
            terminated = True
            self.game_over = True
        elif self.lives <= 0:
            # sfx: game over sound
            reward += self.PENALTY_LOSS
            self.score += self.PENALTY_LOSS
            terminated = True
            self.game_over = True
        elif self.steps >= self.MAX_STEPS:
            terminated = True
            self.game_over = True
            
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _get_min_dist_to_target(self, pos, targets):
        if not targets:
            return float('inf')
        return min(abs(pos[0] - t[0]) + abs(pos[1] - t[1]) for t in targets)

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        if self.game_over:
            self._render_game_over()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw grid
        for x in range(self.GRID_WIDTH):
            for y in range(self.GRID_HEIGHT):
                rect = (x * self.CELL_SIZE, y * self.CELL_SIZE, self.CELL_SIZE, self.CELL_SIZE)
                pygame.draw.rect(self.screen, self.COLOR_GRID, rect, 1)
        
        # Draw gems
        for gx, gy in self.gem_positions:
            self._draw_diamond(gx, gy, self.COLOR_GEM)

        # Draw enemies
        pulse = (math.sin(self.steps * 0.3) + 1) / 2 # 0 to 1
        enemy_radius = int(self.CELL_SIZE * 0.3 + pulse * 3)
        for enemy in self.enemies:
            ex, ey = enemy['pos']
            center_x = int(ex * self.CELL_SIZE + self.CELL_SIZE / 2)
            center_y = int(ey * self.CELL_SIZE + self.CELL_SIZE / 2)
            pygame.gfxdraw.filled_circle(self.screen, center_x, center_y, enemy_radius, self.COLOR_ENEMY)
            pygame.gfxdraw.aacircle(self.screen, center_x, center_y, enemy_radius, self.COLOR_ENEMY)

        # Draw player
        px, py = self.player_pos
        player_rect = (px * self.CELL_SIZE + 2, py * self.CELL_SIZE + 2, self.CELL_SIZE - 4, self.CELL_SIZE - 4)
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, player_rect, border_radius=3)
        
    def _draw_diamond(self, gx, gy, color):
        cx = gx * self.CELL_SIZE + self.CELL_SIZE // 2
        cy = gy * self.CELL_SIZE + self.CELL_SIZE // 2
        half_w = self.CELL_SIZE // 3
        half_h = self.CELL_SIZE // 2 - 2
        points = [
            (cx, cy - half_h),
            (cx + half_w, cy),
            (cx, cy + half_h),
            (cx - half_w, cy)
        ]
        pygame.gfxdraw.filled_polygon(self.screen, points, color)
        pygame.gfxdraw.aapolygon(self.screen, points, color)
        
    def _render_ui(self):
        # Score
        score_text = self.font_small.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))

        # Lives
        for i in range(self.MAX_LIVES):
            heart_x = self.SCREEN_WIDTH - 25 - (i * 30)
            heart_y = 18
            color = self.COLOR_HEART if i < self.lives else self.COLOR_HEART_EMPTY
            self._draw_heart(self.screen, heart_x, heart_y, color)

    def _draw_heart(self, surface, x, y, color):
        # A simple heart shape using polygons
        points = [
            (x, y - 5), (x + 5, y - 10), (x + 10, y - 5),
            (x, y + 5),
            (x - 10, y - 5), (x - 5, y - 10)
        ]
        pygame.gfxdraw.filled_polygon(surface, points, color)
        pygame.gfxdraw.aapolygon(surface, points, color)
        
    def _render_game_over(self):
        overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 180))
        
        message = "YOU WON!" if self.gems_collected >= self.WIN_CONDITION_GEMS else "GAME OVER"
        text = self.font_large.render(message, True, self.COLOR_TEXT)
        text_rect = text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
        self.screen.blit(overlay, (0, 0))
        self.screen.blit(text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "lives": self.lives,
            "gems_collected": self.gems_collected,
        }
        
    def close(self):
        pygame.quit()

    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation.
        '''
        print("Running implementation validation...")
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space
        # We need to reset first to initialize everything for observation
        _, _ = self.reset()
        test_obs = self._get_observation()
        assert test_obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), f"Obs shape is {test_obs.shape}"
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
    env = GameEnv()
    obs, info = env.reset()
    
    running = True
    total_reward = 0
    
    # Game loop
    while running:
        action = [0, 0, 0] # Default action: no-op
        
        # Human controls are event-based for this turn-based game
        move_made = False
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                move_made = True
                movement = 0 # no-op
                if event.key == pygame.K_UP: movement = 1
                elif event.key == pygame.K_DOWN: movement = 2
                elif event.key == pygame.K_LEFT: movement = 3
                elif event.key == pygame.K_RIGHT: movement = 4
                elif event.key == pygame.K_r:
                    obs, info = env.reset()
                    total_reward = 0
                    print("--- Game Reset ---")
                    move_made = False # Don't step on reset
                
                # We don't read space/shift but include them for API compliance
                keys = pygame.key.get_pressed()
                space = 1 if keys[pygame.K_SPACE] else 0
                shift = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
                action = [movement, space, shift]

        if move_made:
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward

            if terminated:
                print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}, Steps: {info['steps']}")

        # Always render the current state
        display_obs = np.transpose(obs, (1, 0, 2))
        try:
            display_surface = pygame.display.get_surface()
            if display_surface is None: raise AttributeError
        except (pygame.error, AttributeError):
             display_surface = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))

        surf = pygame.surfarray.make_surface(display_obs)
        display_surface.blit(surf, (0, 0))
        pygame.display.flip()

        if env.game_over:
            pygame.time.wait(2000)
            obs, info = env.reset()
            total_reward = 0
        
        env.clock.tick(30)

    env.close()