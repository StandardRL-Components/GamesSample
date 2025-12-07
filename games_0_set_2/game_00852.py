import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import math
import os
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    """
    A minimalist block-pushing puzzle game where the player must move all blocks
    onto their designated goal tiles within a limited number of moves.
    """
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Use arrow keys to push all blocks in a direction. The goal is to get all blocks onto the matching goal tiles."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A strategic block-pushing puzzle. Push all blocks to their goals within the move limit. Pushing against a wall is a free move."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    GRID_PIXEL_SIZE = 400
    GRID_WIDTH = 10
    GRID_HEIGHT = 10
    TILE_SIZE = GRID_PIXEL_SIZE // GRID_WIDTH
    NUM_BLOCKS = 15
    MOVE_LIMIT = 50
    MAX_EPISODE_STEPS = 500

    # --- Colors ---
    COLOR_BG = (30, 30, 40)
    COLOR_GRID = (50, 50, 60)
    COLOR_TEXT = (220, 220, 230)
    COLOR_WIN = (100, 255, 100)
    COLOR_LOSE = (255, 100, 100)
    BLOCK_PALETTE = [
        (230, 57, 70), (241, 128, 48), (252, 212, 81), (144, 201, 135),
        (69, 173, 168), (29, 132, 181), (168, 195, 219), (234, 112, 112),
        (131, 56, 236), (180, 222, 222), (255, 190, 11), (251, 86, 7),
        (255, 0, 110), (131, 56, 236), (58, 134, 255)
    ]

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
        self.font_main = pygame.font.SysFont("Consolas", 24)
        self.font_big = pygame.font.SysFont("Consolas", 50, bold=True)
        
        # Initialize state variables
        self.blocks = []
        self.goals = []
        self.steps = 0
        self.score = 0.0
        self.moves_left = 0
        self.game_over = False
        
        # This is called to ensure the np_random generator is initialized before use
        # and to set up the initial state for validation.
        self.reset()
        
        # self.validate_implementation() # Optional validation
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0.0
        self.moves_left = self.MOVE_LIMIT
        self.game_over = False
        
        self._generate_puzzle()
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0.0, True, False, self._get_info()

        # Unpack factorized action
        movement = action[0]
        direction = self._get_direction_from_action(movement)
        
        reward = 0.0
        
        if direction != (0, 0):
            blocks_on_goal_before = self._count_blocks_on_goal()
            
            move_was_legal = self._handle_push(direction)
            
            if move_was_legal:
                self.moves_left -= 1
                reward -= 0.1
                
                blocks_on_goal_after = self._count_blocks_on_goal()
                newly_on_goal = blocks_on_goal_after - blocks_on_goal_before
                if newly_on_goal > 0:
                    reward += newly_on_goal * 1.0
        
        self.steps += 1
        self.score += reward
        
        terminated = self._check_termination()
        truncated = False
        if self.steps >= self.MAX_EPISODE_STEPS:
            truncated = True
            self.game_over = True
        
        if terminated and not self.game_over:
            if self._count_blocks_on_goal() == self.NUM_BLOCKS:
                win_bonus = 50.0
                reward += win_bonus
                self.score += win_bonus
            self.game_over = True
            
        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )
    
    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "moves_left": self.moves_left,
            "blocks_on_goal": self._count_blocks_on_goal(),
        }

    def _generate_puzzle(self):
        self.blocks = []
        self.goals = []
        
        all_pos = [(x, y) for x in range(self.GRID_WIDTH) for y in range(self.GRID_HEIGHT)]
        goal_indices = self.np_random.choice(len(all_pos), size=self.NUM_BLOCKS, replace=False)
        goal_positions = [all_pos[i] for i in goal_indices]
        
        colors = self.np_random.permutation(self.BLOCK_PALETTE)
        for i in range(self.NUM_BLOCKS):
            pos = goal_positions[i]
            color = colors[i % len(colors)]
            self.goals.append({'pos': pos, 'color': color})
            self.blocks.append({'pos': list(pos), 'goal_pos': pos, 'color': color})
            
        num_scrambles = self.np_random.integers(30, 51)
        for _ in range(num_scrambles):
            random_action = self.np_random.integers(1, 5)
            direction = self._get_direction_from_action(random_action)
            self._handle_push(direction)

    def _get_direction_from_action(self, movement_action):
        if movement_action == 1: return (0, -1)  # Up
        if movement_action == 2: return (0, 1)   # Down
        if movement_action == 3: return (-1, 0)  # Left
        if movement_action == 4: return (1, 0)   # Right
        return (0, 0)

    def _handle_push(self, direction):
        dx, dy = direction
        if dx == 0 and dy == 0:
            return False

        any_block_moved = False
        
        if dx > 0: sorted_blocks = sorted(self.blocks, key=lambda b: b['pos'][0], reverse=True)
        elif dx < 0: sorted_blocks = sorted(self.blocks, key=lambda b: b['pos'][0])
        elif dy > 0: sorted_blocks = sorted(self.blocks, key=lambda b: b['pos'][1], reverse=True)
        else: sorted_blocks = sorted(self.blocks, key=lambda b: b['pos'][1])

        block_pos_set = {tuple(b['pos']) for b in self.blocks}

        for block in sorted_blocks:
            current_pos = tuple(block['pos'])
            target_pos = (current_pos[0] + dx, current_pos[1] + dy)

            if not (0 <= target_pos[0] < self.GRID_WIDTH and 0 <= target_pos[1] < self.GRID_HEIGHT):
                continue

            if target_pos in block_pos_set:
                continue

            any_block_moved = True
            block_pos_set.remove(current_pos)
            block_pos_set.add(target_pos)
            block['pos'][0], block['pos'][1] = target_pos[0], target_pos[1]
            
        return any_block_moved

    def _count_blocks_on_goal(self):
        return sum(1 for b in self.blocks if tuple(b['pos']) == b['goal_pos'])

    def _is_stalemate(self):
        for d in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
            if self._can_any_block_move(d):
                return False
        return True

    def _can_any_block_move(self, direction):
        dx, dy = direction
        block_pos_set = {tuple(b['pos']) for b in self.blocks}
        for pos in block_pos_set:
            target_pos = (pos[0] + dx, pos[1] + dy)
            if not (0 <= target_pos[0] < self.GRID_WIDTH and 0 <= target_pos[1] < self.GRID_HEIGHT):
                continue
            if target_pos in block_pos_set:
                continue
            return True
        return False

    def _check_termination(self):
        if self._count_blocks_on_goal() == self.NUM_BLOCKS: return True
        if self.moves_left <= 0: return True
        if self._is_stalemate(): return True
        return False

    def _render_game(self):
        for x in range(self.GRID_WIDTH + 1):
            px = x * self.TILE_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (px, 0), (px, self.GRID_PIXEL_SIZE), 1)
        for y in range(self.GRID_HEIGHT + 1):
            py = y * self.TILE_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, py), (self.GRID_PIXEL_SIZE, py), 1)

        for goal in self.goals:
            gx, gy = goal['pos']
            rect = pygame.Rect(gx * self.TILE_SIZE, gy * self.TILE_SIZE, self.TILE_SIZE, self.TILE_SIZE)
            goal_surface = pygame.Surface((self.TILE_SIZE, self.TILE_SIZE), pygame.SRCALPHA)
            # FIX: Convert numpy array color to tuple before concatenating alpha.
            color_rgba = tuple(goal['color']) + (60,)
            pygame.draw.rect(goal_surface, color_rgba, (2, 2, self.TILE_SIZE - 4, self.TILE_SIZE - 4), border_radius=4)
            self.screen.blit(goal_surface, rect.topleft)

        for block in self.blocks:
            bx, by = block['pos']
            color = block['color']
            rect = pygame.Rect(bx * self.TILE_SIZE + 4, by * self.TILE_SIZE + 4, self.TILE_SIZE - 8, self.TILE_SIZE - 8)
            pygame.draw.rect(self.screen, color, rect, border_radius=6)
            border_color = tuple(max(0, c - 40) for c in color)
            pygame.draw.rect(self.screen, border_color, rect, width=2, border_radius=6)

    def _render_ui(self):
        ui_x = self.GRID_PIXEL_SIZE + 30
        
        moves_text = self.font_main.render(f"MOVES: {self.moves_left}/{self.MOVE_LIMIT}", True, self.COLOR_TEXT)
        self.screen.blit(moves_text, (ui_x, 20))
        
        score_text = self.font_main.render(f"SCORE: {self.score:.1f}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (ui_x, 50))
        
        on_goal = self._count_blocks_on_goal()
        goal_text = self.font_main.render(f"GOALS: {on_goal}/{self.NUM_BLOCKS}", True, self.COLOR_TEXT)
        self.screen.blit(goal_text, (ui_x, 80))
        
        if self.game_over:
            is_win = self._count_blocks_on_goal() == self.NUM_BLOCKS
            msg = "YOU WIN!" if is_win else "GAME OVER"
            color = self.COLOR_WIN if is_win else self.COLOR_LOSE
            
            text_surf = self.font_big.render(msg, True, color)
            text_rect = text_surf.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            
            s = pygame.Surface((text_rect.width + 20, text_rect.height + 20), pygame.SRCALPHA)
            s.fill((0, 0, 0, 180))
            self.screen.blit(s, (text_rect.left - 10, text_rect.top - 10))
            
            self.screen.blit(text_surf, text_rect)

    def close(self):
        pygame.quit()

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
        assert isinstance(trunc, bool)
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == "__main__":
    # This block is for local testing and visualization.
    # It will not be executed by the verification script.
    # To run, you might need to remove the "dummy" video driver setting.
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]

    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    pygame.display.set_caption("GridShift Puzzle")
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    
    running = True
    print(GameEnv.user_guide)
    print("Press 'R' to reset the puzzle.")
    
    while running:
        action = np.array([0, 0, 0])
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_UP: action[0] = 1
                elif event.key == pygame.K_DOWN: action[0] = 2
                elif event.key == pygame.K_LEFT: action[0] = 3
                elif event.key == pygame.K_RIGHT: action[0] = 4
                elif event.key == pygame.K_r:
                    obs, info = env.reset()
                    done = False
                    print("--- Puzzle Reset ---")
                
                if action[0] != 0:
                    if not done:
                        obs, reward, terminated, truncated, info = env.step(action)
                        done = terminated or truncated
                        print(f"Action: {action[0]}, Reward: {reward:.2f}, Done: {done}, Info: {info}")
                    else:
                        print("Game is over. Press 'R' to reset.")

        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        env.clock.tick(30)
        
    env.close()