
# Generated: 2025-08-28T01:38:54.216622
# Source Brief: brief_04184.md
# Brief Index: 4184

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import pygame


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Arrow keys to move one square at a time. "
        "Space and Shift have no effect."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Escape hordes of zombies on a grid-based map to reach a rescue "
        "helicopter while collecting vital supplies. Collect 10 supplies then "
        "reach the chopper to win."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False

    # --- Constants ---
    GRID_WIDTH, GRID_HEIGHT = 32, 20
    CELL_SIZE = 20
    SCREEN_WIDTH, SCREEN_HEIGHT = GRID_WIDTH * CELL_SIZE, GRID_HEIGHT * CELL_SIZE

    COLOR_BG = (44, 62, 80)
    COLOR_GRID = (52, 73, 94)
    COLOR_PLAYER = (46, 204, 113)
    COLOR_PLAYER_GLOW = (46, 204, 113, 60)
    COLOR_ZOMBIE = (231, 76, 60)
    COLOR_SUPPLY = (241, 196, 15)
    COLOR_HELI_PAD = (127, 140, 141)
    COLOR_HELI_H = (236, 240, 241)
    COLOR_BLOOD = (139, 0, 0)
    COLOR_UI_TEXT = (236, 240, 241)

    NUM_ZOMBIES = 5
    NUM_SUPPLIES = 15
    WIN_SUPPLY_COUNT = 10
    MAX_STEPS = 1000

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # EXACT spaces:
        self.observation_space = Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("monospace", 18, bold=True)
        self.font_heli = pygame.font.SysFont("sans-serif", 20, bold=True)
        self.font_game_over = pygame.font.SysFont("monospace", 50, bold=True)

        # Initialize state variables
        self.player_pos = (0, 0)
        self.zombies = []
        self.supplies = []
        self.heli_pos = (0, 0)
        self.supplies_collected = 0
        self.blood_splatters = []
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win_message = ""
        
        # This will be set in reset()
        self.np_random = None

        self.reset()
        
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.supplies_collected = 0
        self.game_over = False
        self.win_message = ""
        self.blood_splatters = []

        # Place player and helicopter
        self.player_pos = (1, 1)
        self.heli_pos = (self.GRID_WIDTH - 2, self.GRID_HEIGHT - 2)

        # Generate all possible grid positions
        all_pos = set((x, y) for x in range(self.GRID_WIDTH) for y in range(self.GRID_HEIGHT))
        all_pos.remove(self.player_pos)
        all_pos.remove(self.heli_pos)

        # Place supplies
        self.supplies = list(self.np_random.choice(list(all_pos), size=self.NUM_SUPPLIES, replace=False))
        self.supplies = [tuple(pos) for pos in self.supplies]
        for pos in self.supplies:
            all_pos.remove(pos)

        # Place zombies at a distance
        self.zombies = []
        possible_zombie_pos = [p for p in all_pos if abs(p[0] - self.player_pos[0]) + abs(p[1] - self.player_pos[1]) > 10]
        
        zombie_indices = self.np_random.choice(len(possible_zombie_pos), size=self.NUM_ZOMBIES, replace=False)
        self.zombies = [possible_zombie_pos[i] for i in zombie_indices]

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # Unpack factorized action
        movement = action[0]  # 0-4: none/up/down/left/right
        
        reward = 0
        terminated = False
        
        # --- Player Movement ---
        px, py = self.player_pos
        prev_player_pos = self.player_pos

        if movement == 1: py -= 1  # Up
        elif movement == 2: py += 1  # Down
        elif movement == 3: px -= 1  # Left
        elif movement == 4: px += 1  # Right
        
        # Clamp to grid
        px = max(0, min(self.GRID_WIDTH - 1, px))
        py = max(0, min(self.GRID_HEIGHT - 1, py))
        self.player_pos = (px, py)

        # --- Zombie Movement ---
        new_zombie_positions = []
        player_x, player_y = self.player_pos
        for zx, zy in self.zombies:
            dx = player_x - zx
            dy = player_y - zy
            
            if dx == 0 and dy == 0:
                new_zombie_positions.append((zx, zy))
                continue

            if abs(dx) > abs(dy):
                zx += int(np.sign(dx))
            else:
                zy += int(np.sign(dy))
            new_zombie_positions.append((zx, zy))
        self.zombies = new_zombie_positions

        # --- Interactions and Rewards ---
        # Supply collection
        if self.player_pos in self.supplies:
            self.supplies.remove(self.player_pos)
            self.supplies_collected += 1
            self.score += 1
            reward += 1.0
            # SFX: supply_pickup

        # Zombie collision
        if self.player_pos in self.zombies:
            reward = -100.0
            self.score -= 100
            terminated = True
            self.game_over = True
            self.win_message = "GAME OVER"
            self.blood_splatters.append(self.player_pos)
            # SFX: player_death

        # Helicopter escape
        if not terminated and self.player_pos == self.heli_pos:
            if self.supplies_collected >= self.WIN_SUPPLY_COUNT:
                reward = 100.0
                self.score += 100
                terminated = True
                self.game_over = True
                self.win_message = "YOU ESCAPED!"
                # SFX: win_jingle
            else:
                # Penalty for reaching heli without enough supplies
                reward -= 0.5 

        # Step penalty/reward based on distance to helicopter
        dist_before = abs(prev_player_pos[0] - self.heli_pos[0]) + abs(prev_player_pos[1] - self.heli_pos[1])
        dist_after = abs(self.player_pos[0] - self.heli_pos[0]) + abs(self.player_pos[1] - self.heli_pos[1])
        if dist_after < dist_before:
            reward += 0.01
        elif dist_after > dist_before:
            reward -= 0.02

        # --- Termination by steps ---
        self.steps += 1
        if not terminated and self.steps >= self.MAX_STEPS:
            terminated = True
            self.game_over = True
            self.win_message = "OUT OF TIME"

        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw grid
        for x in range(0, self.SCREEN_WIDTH, self.CELL_SIZE):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.SCREEN_HEIGHT))
        for y in range(0, self.SCREEN_HEIGHT, self.CELL_SIZE):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.SCREEN_WIDTH, y))

        # Draw helicopter pad
        hx, hy = self.heli_pos
        heli_rect = pygame.Rect(hx * self.CELL_SIZE, hy * self.CELL_SIZE, self.CELL_SIZE, self.CELL_SIZE)
        pygame.draw.rect(self.screen, self.COLOR_HELI_PAD, heli_rect)
        h_text = self.font_heli.render("H", True, self.COLOR_HELI_H)
        text_rect = h_text.get_rect(center=heli_rect.center)
        self.screen.blit(h_text, text_rect)

        # Draw blood splatters
        for bx, by in self.blood_splatters:
            pygame.gfxdraw.filled_circle(
                self.screen,
                int(bx * self.CELL_SIZE + self.CELL_SIZE / 2),
                int(by * self.CELL_SIZE + self.CELL_SIZE / 2),
                int(self.CELL_SIZE * 0.4),
                self.COLOR_BLOOD
            )
            
        # Draw supplies
        for sx, sy in self.supplies:
            pygame.gfxdraw.filled_circle(
                self.screen,
                int(sx * self.CELL_SIZE + self.CELL_SIZE / 2),
                int(sy * self.CELL_SIZE + self.CELL_SIZE / 2),
                int(self.CELL_SIZE * 0.3),
                self.COLOR_SUPPLY
            )
            pygame.gfxdraw.aacircle(
                self.screen,
                int(sx * self.CELL_SIZE + self.CELL_SIZE / 2),
                int(sy * self.CELL_SIZE + self.CELL_SIZE / 2),
                int(self.CELL_SIZE * 0.3),
                self.COLOR_SUPPLY
            )

        # Draw zombies
        for zx, zy in self.zombies:
            zombie_rect = pygame.Rect(zx * self.CELL_SIZE, zy * self.CELL_SIZE, self.CELL_SIZE, self.CELL_SIZE)
            pygame.draw.rect(self.screen, self.COLOR_ZOMBIE, zombie_rect.inflate(-4, -4))

        # Draw player
        px, py = self.player_pos
        player_center_x = int(px * self.CELL_SIZE + self.CELL_SIZE / 2)
        player_center_y = int(py * self.CELL_SIZE + self.CELL_SIZE / 2)
        
        # Glow effect
        pygame.gfxdraw.filled_circle(
            self.screen, player_center_x, player_center_y,
            int(self.CELL_SIZE * 0.75), self.COLOR_PLAYER_GLOW
        )
        
        player_rect = pygame.Rect(px * self.CELL_SIZE, py * self.CELL_SIZE, self.CELL_SIZE, self.CELL_SIZE)
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, player_rect.inflate(-4, -4))


    def _render_ui(self):
        # Supplies text
        supply_text = self.font_ui.render(
            f"Supplies: {self.supplies_collected}/{self.WIN_SUPPLY_COUNT}", True, self.COLOR_UI_TEXT
        )
        self.screen.blit(supply_text, (10, 10))

        # Steps text
        step_text = self.font_ui.render(
            f"Step: {self.steps}/{self.MAX_STEPS}", True, self.COLOR_UI_TEXT
        )
        step_rect = step_text.get_rect(topright=(self.SCREEN_WIDTH - 10, 10))
        self.screen.blit(step_text, step_rect)
        
        # Game Over message
        if self.game_over:
            over_text = self.font_game_over.render(self.win_message, True, self.COLOR_UI_TEXT)
            over_rect = over_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            
            # Create a semi-transparent background for the text
            s = pygame.Surface(over_rect.size, pygame.SRCALPHA)
            s.fill((0,0,0,128))
            self.screen.blit(s, over_rect.topleft)
            
            self.screen.blit(over_text, over_rect)


    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "supplies_collected": self.supplies_collected,
            "player_pos": self.player_pos,
            "zombie_count": len(self.zombies),
        }

    def close(self):
        pygame.quit()
        
    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation:
        '''
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
        assert self.player_pos[0] >= 0 and self.player_pos[0] < self.GRID_WIDTH
        assert self.player_pos[1] >= 0 and self.player_pos[1] < self.GRID_HEIGHT
        assert len(self.supplies) == self.NUM_SUPPLIES
        assert len(self.zombies) == self.NUM_ZOMBIES
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

# Example of how to run the environment
if __name__ == "__main__":
    import os
    os.environ["SDL_VIDEODRIVER"] = "dummy" # Run headless
    
    env = GameEnv(render_mode="rgb_array")
    
    # --- To play manually (requires a display) ---
    # To run this part, comment out the os.environ line above
    # and install pygame: pip install pygame
    #
    # env = GameEnv(render_mode="rgb_array")
    # obs, info = env.reset()
    # done = False
    # screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    # pygame.display.set_caption("Zombie Grid Escape")
    # clock = pygame.time.Clock()
    
    # running = True
    # while running:
    #     action = [0, 0, 0] # Default action: no-op
    #     for event in pygame.event.get():
    #         if event.type == pygame.QUIT:
    #             running = False
    #         if event.type == pygame.KEYDOWN:
    #             if event.key == pygame.K_UP:
    #                 action[0] = 1
    #             elif event.key == pygame.K_DOWN:
    #                 action[0] = 2
    #             elif event.key == pygame.K_LEFT:
    #                 action[0] = 3
    #             elif event.key == pygame.K_RIGHT:
    #                 action[0] = 4
    #             elif event.key == pygame.K_r: # Reset
    #                 obs, info = env.reset()
    #                 done = False
    #             elif event.key == pygame.K_q:
    #                 running = False
        
    #     if not done:
    #         obs, reward, terminated, truncated, info = env.step(action)
    #         done = terminated or truncated
    #         print(f"Step: {info['steps']}, Reward: {reward:.2f}, Score: {info['score']}, Done: {done}")

    #     # Display the observation
    #     frame = np.transpose(obs, (1, 0, 2))
    #     surf = pygame.surfarray.make_surface(frame)
    #     screen.blit(surf, (0, 0))
    #     pygame.display.flip()
        
    #     clock.tick(10) # Control speed of manual play

    # env.close()
    # pygame.quit()