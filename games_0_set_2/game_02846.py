
# Generated: 2025-08-27T21:36:22.754680
# Source Brief: brief_02846.md
# Brief Index: 2846

        
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
    """
    An arcade-style grid survival game where the player collects coins and dodges monsters.
    The game is turn-based, with the world state updating after each player action.
    """
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Use arrow keys to move on the grid. Collect gold coins to score points."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Navigate a grid, dodging procedurally generated monsters while collecting coins to amass a fortune."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False
    
    def __init__(self, render_mode="rgb_array"):
        """
        Initializes the game environment, including Pygame, spaces, and game constants.
        """
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
        self.GRID_WIDTH = 20
        self.GRID_HEIGHT = 15
        self.CELL_SIZE = 20
        self.GAME_AREA_WIDTH = self.GRID_WIDTH * self.CELL_SIZE  # 400
        self.GAME_AREA_HEIGHT = self.GRID_HEIGHT * self.CELL_SIZE # 300
        self.X_OFFSET = (640 - self.GAME_AREA_WIDTH) // 2 # 120
        self.Y_OFFSET = (400 - self.GAME_AREA_HEIGHT) // 2 # 50
        self.MAX_STEPS = 1000
        self.WIN_COIN_COUNT = 100
        self.MAX_MONSTER_TOUCHES = 5
        self.INITIAL_MONSTERS = 3
        self.COINS_ON_SCREEN = 5

        # Colors
        self.COLOR_BG = (20, 25, 40)
        self.COLOR_GRID = (40, 50, 70)
        self.COLOR_PLAYER = (255, 255, 0)
        self.COLOR_MONSTER = (255, 50, 50)
        self.COLOR_COIN = (255, 215, 0)
        self.COLOR_UI_TEXT = (240, 240, 240)
        self.COLOR_UI_BG = (10, 10, 20, 180) # RGBA for transparency

        # Fonts
        self.font_ui = pygame.font.Font(None, 24)
        self.font_game_over = pygame.font.Font(None, 72)
        
        # State variables (initialized in reset)
        self.player_pos = None
        self.monster_pos = []
        self.coin_pos = []
        self.steps = 0
        self.score = 0
        self.coins_collected = 0
        self.monsters_touched = 0
        self.game_over = False
        self.monster_move_debt = 0.0
        
        self.reset()
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        """
        Resets the game state to its initial configuration.
        """
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.coins_collected = 0
        self.monsters_touched = 0
        self.game_over = False
        self.monster_move_debt = 0.0
        
        self._place_elements()
        
        return self._get_observation(), self._get_info()

    def _place_elements(self):
        """
        Procedurally places the player, monsters, and coins on the grid without overlap.
        """
        occupied_cells = set()
        
        # Player
        self.player_pos = self._get_random_empty_cell(occupied_cells)
        occupied_cells.add(self.player_pos)
        
        # Monsters
        self.monster_pos = []
        for _ in range(self.INITIAL_MONSTERS):
            pos = self._get_random_empty_cell(occupied_cells)
            self.monster_pos.append(pos)
            occupied_cells.add(pos)
            
        # Coins
        self.coin_pos = []
        for _ in range(self.COINS_ON_SCREEN):
            pos = self._get_random_empty_cell(occupied_cells)
            self.coin_pos.append(pos)
            occupied_cells.add(pos)

    def _get_random_empty_cell(self, occupied_cells):
        """
        Finds a random, unoccupied cell on the grid.
        """
        while True:
            pos = (
                self.np_random.integers(0, self.GRID_HEIGHT),
                self.np_random.integers(0, self.GRID_WIDTH)
            )
            if pos not in occupied_cells:
                return pos
    
    def step(self, action):
        """
        Advances the game state by one turn based on the player's action.
        """
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement = action[0]
        reward = 0.0
        
        # 1. Update Player Position
        player_moved = self._move_player(movement)
        if player_moved:
            reward = -0.2  # Cost for moving

        # 2. Check for Coin Collection
        if self.player_pos in self.coin_pos:
            # Sound: coin_collect.wav
            reward += 1.2  # Net reward of +1.0 for collecting a coin
            self.score += 1
            self.coins_collected += 1
            self.coin_pos.remove(self.player_pos)
            
            # Spawn a new coin if not about to win
            if self.coins_collected < self.WIN_COIN_COUNT:
                occupied_cells = {self.player_pos, *self.monster_pos, *self.coin_pos}
                new_coin_pos = self._get_random_empty_cell(occupied_cells)
                self.coin_pos.append(new_coin_pos)

        # 3. Update Monster Positions
        self._move_monsters()
        
        # 4. Check for Monster Collision
        if self.player_pos in self.monster_pos:
            # Sound: player_hit.wav
            self.monsters_touched += 1
        
        # 5. Update game state
        self.steps += 1
        
        # 6. Check for termination and apply terminal rewards
        terminated = self._check_termination()
        if terminated:
            self.game_over = True
            if self.coins_collected >= self.WIN_COIN_COUNT:
                # Sound: game_win.wav
                reward += 105.0  # +100 for win, +5 for last coin
            elif self.monsters_touched >= self.MAX_MONSTER_TOUCHES:
                # Sound: game_over.wav
                reward -= 100.0  # -100 for loss
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )

    def _move_player(self, movement):
        """Moves the player based on the action, handling grid boundaries."""
        row, col = self.player_pos
        prev_pos = self.player_pos
        
        if movement == 1: row -= 1  # Up
        elif movement == 2: row += 1  # Down
        elif movement == 3: col -= 1  # Left
        elif movement == 4: col += 1  # Right
        
        # Boundary check
        if 0 <= row < self.GRID_HEIGHT and 0 <= col < self.GRID_WIDTH:
            self.player_pos = (row, col)
            
        return self.player_pos != prev_pos

    def _move_monsters(self):
        """Moves each monster randomly to an adjacent, valid cell."""
        # Difficulty scaling: monsters get faster over time
        num_moves_float = 1.0 + 0.05 * math.floor(self.steps / 200)
        self.monster_move_debt += num_moves_float
        num_moves_int = math.floor(self.monster_move_debt)
        if num_moves_int == 0:
            return
        self.monster_move_debt -= num_moves_int

        for _ in range(num_moves_int):
            # Sound: monster_step.wav
            new_monster_positions = []
            for i, pos in enumerate(self.monster_pos):
                # Monsters avoid each other and the player
                occupied_cells = {self.player_pos, *new_monster_positions, *self.monster_pos[i+1:]}
                
                row, col = pos
                possible_moves = []
                for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                    new_row, new_col = row + dr, col + dc
                    if 0 <= new_row < self.GRID_HEIGHT and 0 <= new_col < self.GRID_WIDTH:
                        if (new_row, new_col) not in occupied_cells:
                            possible_moves.append((new_row, new_col))
                
                if possible_moves:
                    move_idx = self.np_random.integers(0, len(possible_moves))
                    new_monster_positions.append(possible_moves[move_idx])
                else:
                    new_monster_positions.append(pos) # Stay still if blocked
            self.monster_pos = new_monster_positions

    def _check_termination(self):
        """Checks if the game has reached a terminal state."""
        if self.coins_collected >= self.WIN_COIN_COUNT:
            return True
        if self.monsters_touched >= self.MAX_MONSTER_TOUCHES:
            return True
        if self.steps >= self.MAX_STEPS:
            return True
        return False

    def _get_observation(self):
        """Renders the current game state to a Pygame surface and returns it as a numpy array."""
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        
        # Convert to numpy array (EXACT format required)
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        """Renders the grid and all game entities."""
        # Draw grid lines
        for row in range(self.GRID_HEIGHT + 1):
            y = self.Y_OFFSET + row * self.CELL_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (self.X_OFFSET, y), (self.X_OFFSET + self.GAME_AREA_WIDTH, y))
        for col in range(self.GRID_WIDTH + 1):
            x = self.X_OFFSET + col * self.CELL_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, self.Y_OFFSET), (x, self.Y_OFFSET + self.GAME_AREA_HEIGHT))

        # Draw coins (gold circles)
        for row, col in self.coin_pos:
            cx = self.X_OFFSET + col * self.CELL_SIZE + self.CELL_SIZE // 2
            cy = self.Y_OFFSET + row * self.CELL_SIZE + self.CELL_SIZE // 2
            radius = self.CELL_SIZE // 3
            pygame.gfxdraw.filled_circle(self.screen, cx, cy, radius, self.COLOR_COIN)
            pygame.gfxdraw.aacircle(self.screen, cx, cy, radius, self.COLOR_COIN)

        # Draw monsters (red circles)
        for row, col in self.monster_pos:
            cx = self.X_OFFSET + col * self.CELL_SIZE + self.CELL_SIZE // 2
            cy = self.Y_OFFSET + row * self.CELL_SIZE + self.CELL_SIZE // 2
            radius = self.CELL_SIZE // 2 - 2
            pygame.gfxdraw.filled_circle(self.screen, cx, cy, radius, self.COLOR_MONSTER)
            pygame.gfxdraw.aacircle(self.screen, cx, cy, radius, self.COLOR_MONSTER)
        
        # Draw player (yellow square)
        if self.player_pos:
            row, col = self.player_pos
            player_rect = pygame.Rect(
                self.X_OFFSET + col * self.CELL_SIZE + 2,
                self.Y_OFFSET + row * self.CELL_SIZE + 2,
                self.CELL_SIZE - 4,
                self.CELL_SIZE - 4
            )
            pygame.draw.rect(self.screen, self.COLOR_PLAYER, player_rect)

    def _render_ui(self):
        """Renders the UI elements like score and game status."""
        # Helper to render text with a background
        def render_text_with_bg(text, pos, font):
            text_surf = font.render(text, True, self.COLOR_UI_TEXT)
            bg_rect = text_surf.get_rect(topleft=pos)
            bg_rect.inflate_ip(10, 6) # Add padding
            
            bg_surf = pygame.Surface(bg_rect.size, pygame.SRCALPHA)
            bg_surf.fill(self.COLOR_UI_BG)
            self.screen.blit(bg_surf, bg_rect)
            self.screen.blit(text_surf, (bg_rect.x + 5, bg_rect.y + 3))

        render_text_with_bg(f"Score: {self.score}", (10, 10), self.font_ui)
        render_text_with_bg(f"Coins: {self.coins_collected}/{self.WIN_COIN_COUNT}", (500, 10), self.font_ui)
        render_text_with_bg(f"Hits: {self.monsters_touched}/{self.MAX_MONSTER_TOUCHES}", (10, 360), self.font_ui)
        
        if self.game_over:
            message = "YOU WIN!" if self.coins_collected >= self.WIN_COIN_COUNT else "GAME OVER"
            color = (100, 255, 100) if self.coins_collected >= self.WIN_COIN_COUNT else (255, 100, 100)
            
            text_surf = self.font_game_over.render(message, True, color)
            text_rect = text_surf.get_rect(center=(640 // 2, 400 // 2))
            
            # Draw a shadow
            shadow_surf = self.font_game_over.render(message, True, (0,0,0))
            self.screen.blit(shadow_surf, text_rect.move(3, 3))
            
            self.screen.blit(text_surf, text_rect)

    def _get_info(self):
        """Returns a dictionary with auxiliary diagnostic information."""
        return {
            "score": self.score,
            "steps": self.steps,
            "coins_collected": self.coins_collected,
            "monsters_touched": self.monsters_touched
        }

    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation.
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

if __name__ == '__main__':
    # This block allows you to play the game directly
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # Setup Pygame window for human play
    pygame.display.set_caption("Grid Survival")
    screen = pygame.display.set_mode((640, 400))
    clock = pygame.time.Clock()
    
    running = True
    while running:
        movement = 0 # No-op by default
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]:
            movement = 1
        elif keys[pygame.K_DOWN]:
            movement = 2
        elif keys[pygame.K_LEFT]:
            movement = 3
        elif keys[pygame.K_RIGHT]:
            movement = 4
        
        # The game is not auto-advancing, so we need to step on an input
        # We'll step once per frame for human play, even with no-op
        action = [movement, 0, 0] # Space and shift are not used
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Render the observation to the display window
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated:
            print(f"Game Over! Final Score: {info['score']}, Steps: {info['steps']}")
            pygame.time.wait(3000) # Pause for 3 seconds
            obs, info = env.reset()

        clock.tick(10) # Control human play speed

    pygame.quit()