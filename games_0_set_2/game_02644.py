
# Generated: 2025-08-27T20:59:46.037783
# Source Brief: brief_02644.md
# Brief Index: 2644

        
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
        "Controls: Use arrow keys to move your character (red circle). "
        "Push the brown crates onto the green targets."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A fast-paced Sokoban variant. Race against the clock to push all "
        "crates onto the target locations before time runs out."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.GRID_SIZE = (16, 10)
        self.TILE_SIZE = 40
        self.FPS = 30
        self.GAME_DURATION_SECONDS = 60
        self.MAX_STEPS = self.GAME_DURATION_SECONDS * self.FPS
        self.MOVE_COOLDOWN_FRAMES = 5 # Player can move once every 5 steps

        # --- Colors ---
        self.COLOR_BG = (20, 30, 40)
        self.COLOR_WALL = (70, 80, 90)
        self.COLOR_GRID = (30, 40, 50)
        self.COLOR_TARGET = (40, 100, 60)
        self.COLOR_TARGET_LIT = (60, 150, 90)
        self.COLOR_CRATE = (139, 69, 19)
        self.COLOR_CRATE_ON_TARGET = (180, 120, 50)
        self.COLOR_PLAYER = (220, 50, 50)
        self.COLOR_PLAYER_GLOW = (255, 100, 100)
        self.COLOR_UI_TEXT = (230, 230, 230)
        self.COLOR_UI_BG = (10, 10, 10, 180) # Semi-transparent

        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("monospace", 20, bold=True)
        self.font_game_over = pygame.font.SysFont("monospace", 48, bold=True)

        # Game state variables are initialized in reset()
        self.player_pos = None
        self.crate_pos = None
        self.target_pos = None
        self.wall_map = None
        self.steps = None
        self.score = None
        self.timer = None
        self.game_over = None
        self.win = None
        self.move_cooldown = None
        self.crates_on_target_indices = None
        
        # Initialize state variables
        self.reset()
        
        # Self-check
        # self.validate_implementation() # Uncomment for debugging

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.timer = float(self.GAME_DURATION_SECONDS)
        self.game_over = False
        self.win = False
        self.move_cooldown = 0
        
        self._generate_level()

        return self._get_observation(), self._get_info()

    def _generate_level(self):
        # Using a fixed, solvable level design for consistency
        self.wall_map = set()
        for x in range(self.GRID_SIZE[0]):
            self.wall_map.add((x, 0))
            self.wall_map.add((x, self.GRID_SIZE[1] - 1))
        for y in range(self.GRID_SIZE[1]):
            self.wall_map.add((0, y))
            self.wall_map.add((self.GRID_SIZE[0] - 1, y))
        
        # Add some internal obstacles
        for y in range(3, 7):
            self.wall_map.add((4, y))
            self.wall_map.add((11, y))

        self.player_pos = [2, 4]
        self.crate_pos = [[6, 2], [7, 6], [8, 3], [9, 7]]
        self.target_pos = [[13, 2], [13, 4], [13, 6], [13, 8]]

        # Ensure no starting positions overlap with walls
        while tuple(self.player_pos) in self.wall_map:
            self.player_pos[0] += 1
        for i in range(len(self.crate_pos)):
            while tuple(self.crate_pos[i]) in self.wall_map:
                self.crate_pos[i][0] += 1
        
        self.crates_on_target_indices = self._get_crates_on_targets()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement = action[0]
        # space_held = action[1] == 1
        # shift_held = action[2] == 1

        reward = -0.01  # Small penalty for taking time
        
        # Update timer
        self.timer -= 1.0 / self.FPS
        self.steps += 1
        if self.move_cooldown > 0:
            self.move_cooldown -= 1

        # --- Handle player movement ---
        if movement != 0 and self.move_cooldown <= 0:
            self.move_cooldown = self.MOVE_COOLDOWN_FRAMES
            dx, dy = 0, 0
            if movement == 1: dy = -1  # Up
            elif movement == 2: dy = 1  # Down
            elif movement == 3: dx = -1  # Left
            elif movement == 4: dx = 1  # Right
            
            # --- Collision and Push Logic ---
            next_player_pos = [self.player_pos[0] + dx, self.player_pos[1] + dy]
            
            # Check for crate collision
            crate_idx = self._get_crate_at(next_player_pos)
            if crate_idx is not None:
                # Attempt to push the crate
                next_crate_pos = [next_player_pos[0] + dx, next_player_pos[1] + dy]
                # Check if the space behind the crate is empty
                if tuple(next_crate_pos) not in self.wall_map and self._get_crate_at(next_crate_pos) is None:
                    # Move crate and player
                    self.crate_pos[crate_idx] = next_crate_pos
                    self.player_pos = next_player_pos
                    # sfx: push_crate.wav
            
            # Check for wall collision (if not pushing a crate)
            elif tuple(next_player_pos) not in self.wall_map:
                # Move player
                self.player_pos = next_player_pos
                # sfx: player_step.wav

        # --- Calculate rewards for placing crates ---
        new_crates_on_target = self._get_crates_on_targets()
        newly_placed_count = len(new_crates_on_target - self.crates_on_target_indices)
        if newly_placed_count > 0:
            reward += newly_placed_count * 1.0
            # sfx: crate_on_target.wav
        self.crates_on_target_indices = new_crates_on_target
        
        self.score += reward
        terminated = self._check_termination()

        # --- Terminal rewards ---
        if terminated:
            if self.win:
                reward += 100.0
                # sfx: victory.wav
            else: # Time ran out
                reward -= 100.0
                # sfx: failure.wav
            self.score += reward # Add terminal reward to final score

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _get_crate_at(self, pos):
        for i, c_pos in enumerate(self.crate_pos):
            if c_pos[0] == pos[0] and c_pos[1] == pos[1]:
                return i
        return None

    def _get_crates_on_targets(self):
        on_target = set()
        # Use a tolerance for floating point positions if they existed
        # but here we use integer grid coords.
        crate_tuples = {tuple(p) for p in self.crate_pos}
        target_tuples = {tuple(p) for p in self.target_pos}
        
        for i, c_pos in enumerate(self.crate_pos):
            if tuple(c_pos) in target_tuples:
                on_target.add(i)
        return on_target

    def _check_termination(self):
        if self.game_over:
            return True
            
        # Win condition
        if len(self.crates_on_target_indices) == len(self.target_pos):
            self.game_over = True
            self.win = True
            return True
            
        # Lose condition
        if self.timer <= 0:
            self.timer = 0
            self.game_over = True
            self.win = False
            return True
            
        return False

    def _get_observation(self):
        # Clear screen with background
        self.screen.fill(self.COLOR_BG)
        
        # Render all game elements
        self._render_game()
        
        # Render UI overlay
        self._render_ui()
        
        # Convert to numpy array (EXACT format required)
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw grid lines
        for x in range(self.GRID_SIZE[0]):
            for y in range(self.GRID_SIZE[1]):
                rect = pygame.Rect(x * self.TILE_SIZE, y * self.TILE_SIZE, self.TILE_SIZE, self.TILE_SIZE)
                pygame.draw.rect(self.screen, self.COLOR_GRID, rect, 1)

        # Draw targets
        for pos in self.target_pos:
            rect = pygame.Rect(pos[0] * self.TILE_SIZE, pos[1] * self.TILE_SIZE, self.TILE_SIZE, self.TILE_SIZE)
            is_lit = False
            for crate_idx in self.crates_on_target_indices:
                if self.crate_pos[crate_idx] == pos:
                    is_lit = True
                    break
            color = self.COLOR_TARGET_LIT if is_lit else self.COLOR_TARGET
            pygame.draw.rect(self.screen, color, rect)

        # Draw walls
        for pos in self.wall_map:
            rect = pygame.Rect(pos[0] * self.TILE_SIZE, pos[1] * self.TILE_SIZE, self.TILE_SIZE, self.TILE_SIZE)
            pygame.draw.rect(self.screen, self.COLOR_WALL, rect)
            
        # Draw crates
        for i, pos in enumerate(self.crate_pos):
            rect = pygame.Rect(
                pos[0] * self.TILE_SIZE + 4, 
                pos[1] * self.TILE_SIZE + 4, 
                self.TILE_SIZE - 8, 
                self.TILE_SIZE - 8
            )
            color = self.COLOR_CRATE_ON_TARGET if i in self.crates_on_target_indices else self.COLOR_CRATE
            pygame.draw.rect(self.screen, color, rect, border_radius=4)
            pygame.draw.rect(self.screen, tuple(c*0.8 for c in color), rect, 2, border_radius=4)

        # Draw player
        center_x = int(self.player_pos[0] * self.TILE_SIZE + self.TILE_SIZE / 2)
        center_y = int(self.player_pos[1] * self.TILE_SIZE + self.TILE_SIZE / 2)
        radius = int(self.TILE_SIZE / 2 - 4)
        
        # Glow effect
        pygame.gfxdraw.filled_circle(self.screen, center_x, center_y, radius, self.COLOR_PLAYER_GLOW)
        pygame.gfxdraw.aaellipse(self.screen, center_x, center_y, radius, radius, self.COLOR_PLAYER_GLOW)
        # Main circle
        pygame.gfxdraw.filled_circle(self.screen, center_x, center_y, radius - 2, self.COLOR_PLAYER)
        pygame.gfxdraw.aacircle(self.screen, center_x, center_y, radius - 2, self.COLOR_PLAYER)

    def _render_ui(self):
        # UI Panel
        ui_panel_rect = pygame.Rect(self.WIDTH - 200, 10, 190, 70)
        s = pygame.Surface((ui_panel_rect.width, ui_panel_rect.height), pygame.SRCALPHA)
        s.fill(self.COLOR_UI_BG)
        self.screen.blit(s, (ui_panel_rect.x, ui_panel_rect.y))
        pygame.draw.rect(self.screen, self.COLOR_UI_TEXT, ui_panel_rect, 1, 3)

        # Timer display
        timer_text = f"TIME: {max(0, self.timer):.1f}"
        timer_surf = self.font_ui.render(timer_text, True, self.COLOR_UI_TEXT)
        self.screen.blit(timer_surf, (self.WIDTH - 190, 20))

        # Crates on target display
        crates_text = f"GOAL: {len(self.crates_on_target_indices)} / {len(self.target_pos)}"
        crates_surf = self.font_ui.render(crates_text, True, self.COLOR_UI_TEXT)
        self.screen.blit(crates_surf, (self.WIDTH - 190, 50))

        # Game Over message
        if self.game_over:
            message = "VICTORY!" if self.win else "TIME UP!"
            color = (100, 255, 100) if self.win else (255, 100, 100)
            text_surf = self.font_game_over.render(message, True, color)
            text_rect = text_surf.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2))
            
            # Add a background to the text for readability
            bg_rect = text_rect.inflate(20, 20)
            s = pygame.Surface(bg_rect.size, pygame.SRCALPHA)
            s.fill(self.COLOR_UI_BG)
            self.screen.blit(s, bg_rect.topleft)
            
            self.screen.blit(text_surf, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "timer": self.timer,
            "crates_on_target": len(self.crates_on_target_indices),
            "win": self.win,
        }

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation:
        '''
        print("Running implementation validation...")
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == "__main__":
    # This block allows you to play the game manually
    env = GameEnv(render_mode="rgb_array")
    env.reset()
    
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Sokoban Racer")
    clock = pygame.time.Clock()

    running = True
    total_reward = 0.0

    while running:
        movement = 0 # No-op
        space = 0
        shift = 0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                env.reset()
                total_reward = 0.0

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        if keys[pygame.K_SPACE]: space = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift = 1

        action = [movement, space, shift]
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # Render the observation to the display
        # Pygame uses (width, height), numpy uses (height, width)
        # We need to transpose back for display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated:
            print(f"Game Over! Final Score: {info['score']:.2f}, Win: {info['win']}")
            # Wait for a moment before auto-resetting
            pygame.time.wait(3000)
            env.reset()
            total_reward = 0.0

        clock.tick(env.FPS)
    
    env.close()