
# Generated: 2025-08-27T13:03:17.867841
# Source Brief: brief_00242.md
# Brief Index: 242

        
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
    A puzzle game where the player pushes colored blocks around a grid to
    match them with their corresponding targets. Each push action moves all
    unlocked blocks simultaneously in the chosen direction until they hit an
    obstacle. The goal is to solve the puzzle within a limited number of moves.
    """
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Use Arrow Keys (↑↓←→) to push all blocks in a direction."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A strategic puzzle game. Push all blocks simultaneously to slide them "
        "onto their matching colored targets before you run out of moves."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False
    
    def __init__(self, render_mode="rgb_array"):
        """
        Initializes the game environment, including Gymnasium spaces, Pygame,
        and game-specific constants and state variables.
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
        self.GRID_SIZE = (10, 8)  # width, height
        self.CELL_SIZE = 40
        self.GRID_OFFSET_X = (640 - self.GRID_SIZE[0] * self.CELL_SIZE) // 2
        self.GRID_OFFSET_Y = (400 - self.GRID_SIZE[1] * self.CELL_SIZE) // 2
        self.NUM_BLOCKS = 4
        self.MAX_MOVES = 20
        self.SHUFFLE_DEPTH = 15

        # Colors
        self.COLOR_BG = (20, 30, 40)
        self.COLOR_GRID = (40, 50, 60)
        self.BLOCK_COLORS = [
            (255, 87, 34),   # Deep Orange
            (3, 169, 244),   # Light Blue
            (139, 195, 74),  # Light Green
            (255, 235, 59),  # Yellow
        ]
        self.COLOR_TEXT = (224, 224, 224)
        self.COLOR_BONUS = (255, 235, 59)
        self.COLOR_WIN = (100, 255, 100)
        self.COLOR_LOSS = (255, 100, 100)
        
        # Fonts
        self.font_main = pygame.font.Font(None, 36)
        self.font_bonus = pygame.font.Font(None, 28)

        # State variables (to be initialized in reset)
        self.blocks = []
        self.targets = []
        self.moves_remaining = 0
        self.score = 0.0
        self.steps = 0
        self.game_over = False
        self.bonus_text = ""
        self.bonus_text_timer = 0
        self.np_random = None
        
        self.reset()
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        """
        Resets the environment to a new initial state and returns the initial
        observation and info dictionary.
        """
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0.0
        self.game_over = False
        self.moves_remaining = self.MAX_MOVES
        self.bonus_text = ""
        self.bonus_text_timer = 0
        
        self._generate_level()
        
        return self._get_observation(), self._get_info()

    def _generate_level(self):
        """
        Generates a new, solvable puzzle by placing blocks on targets and then
        shuffling them with a series of random pushes.
        """
        while True:
            all_pos = [(x, y) for x in range(self.GRID_SIZE[0]) for y in range(self.GRID_SIZE[1])]
            self.np_random.shuffle(all_pos)
            
            self.targets = []
            self.blocks = []
            for i in range(self.NUM_BLOCKS):
                pos = all_pos.pop()
                color = self.BLOCK_COLORS[i]
                self.targets.append({"pos": pos, "color": color})
                self.blocks.append({"pos": pos, "color": color, "target_pos": pos, "on_target": False})

            for _ in range(self.SHUFFLE_DEPTH):
                direction = self.np_random.integers(1, 5) # 1-4 for directions
                self._execute_push(direction, is_setup=True)
            
            self._update_on_target_status()
            if not self._check_win_condition():
                break # Ensure the shuffled state is not already solved

    def step(self, action):
        """
        Executes one time step in the environment.
        """
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        movement = action[0]  # 0-4: none/up/down/left/right
        reward = 0.0
        
        if self.bonus_text_timer > 0:
            self.bonus_text_timer -= 1
        if self.bonus_text_timer == 0:
            self.bonus_text = ""

        if movement != 0: # 0 is no-op
            self.moves_remaining -= 1
            reward -= 0.1  # Cost of making a move

            pre_move_distances = self._get_all_distances()
            
            risk_bonus_earned = self._execute_push(movement)
            if risk_bonus_earned:
                reward += 2.0
                self.bonus_text = "RISK BONUS +2.0"
                self.bonus_text_timer = 30 # Frames

            post_move_distances = self._get_all_distances()
            
            for i in range(self.NUM_BLOCKS):
                if post_move_distances[i] < pre_move_distances[i]:
                    reward += 1.0 # Reward for moving a block closer

            placement_reward = self._update_on_target_status()
            reward += placement_reward

        self.steps += 1
        self.score += reward
        terminated = self._check_termination()
        
        if terminated:
            if self._check_win_condition():
                reward += 50.0
                self.score += 50.0
            else:
                reward += -50.0
                self.score += -50.0
            self.game_over = True

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _execute_push(self, direction, is_setup=False):
        """
        Handles the core game logic of pushing all blocks.
        """
        # 1=up, 2=down, 3=left, 4=right
        if direction == 1: # Up
            self.blocks.sort(key=lambda b: b['pos'][1])
            delta = (0, -1)
        elif direction == 2: # Down
            self.blocks.sort(key=lambda b: b['pos'][1], reverse=True)
            delta = (0, 1)
        elif direction == 3: # Left
            self.blocks.sort(key=lambda b: b['pos'][0])
            delta = (-1, 0)
        else: # 4 = Right
            self.blocks.sort(key=lambda b: b['pos'][0], reverse=True)
            delta = (1, 0)
            
        grid_w, grid_h = self.GRID_SIZE
        risky_move = False

        for block in self.blocks:
            if block['on_target'] and not is_setup:
                continue

            current_pos = list(block['pos'])
            
            while True:
                next_pos = (current_pos[0] + delta[0], current_pos[1] + delta[1])
                
                if not (0 <= next_pos[0] < grid_w and 0 <= next_pos[1] < grid_h):
                    break # Hit a wall

                if any(other['pos'] == next_pos for other in self.blocks):
                    break # Hit another block
                
                current_pos[0], current_pos[1] = next_pos
            
            block['pos'] = tuple(current_pos)
        
        if not is_setup:
            for block in self.blocks:
                x, y = block['pos']
                if (direction == 1 and y == 0) or \
                   (direction == 2 and y == grid_h - 1) or \
                   (direction == 3 and x == 0) or \
                   (direction == 4 and x == grid_w - 1):
                    risky_move = True
                    break
        
        return risky_move

    def _get_all_distances(self):
        """Calculates Manhattan distance for each block to its target."""
        return [
            abs(b['pos'][0] - b['target_pos'][0]) + abs(b['pos'][1] - b['target_pos'][1])
            for b in self.blocks
        ]
        
    def _update_on_target_status(self):
        """Updates which blocks are on their targets and returns placement rewards."""
        placement_reward = 0.0
        for block in self.blocks:
            is_now_on_target = block['pos'] == block['target_pos']
            if is_now_on_target and not block['on_target']:
                placement_reward += 5.0
                # sfx_block_lock()
                self.bonus_text = "BLOCK PLACED +5.0"
                self.bonus_text_timer = 30
            block['on_target'] = is_now_on_target
        return placement_reward

    def _check_win_condition(self):
        """Checks if all blocks are on their targets."""
        return all(b['on_target'] for b in self.blocks)

    def _check_termination(self):
        """Checks for win, loss, or step limit termination conditions."""
        if self._check_win_condition():
            return True
        if self.moves_remaining <= 0:
            return True
        if self.steps >= 1000:
            return True
        return False
        
    def _get_observation(self):
        """Renders the current game state to a NumPy array."""
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        """Renders the grid, targets, and blocks."""
        gw, gh = self.GRID_SIZE
        cs = self.CELL_SIZE
        ox, oy = self.GRID_OFFSET_X, self.GRID_OFFSET_Y

        # Draw grid lines
        for x in range(gw + 1):
            pygame.draw.line(self.screen, self.COLOR_GRID, (ox + x * cs, oy), (ox + x * cs, oy + gh * cs), 1)
        for y in range(gh + 1):
            pygame.draw.line(self.screen, self.COLOR_GRID, (ox, oy + y * cs), (ox + gw * cs, oy + y * cs), 1)

        # Draw targets
        for target in self.targets:
            x, y = target['pos']
            rect = pygame.Rect(ox + x * cs, oy + y * cs, cs, cs)
            pygame.draw.rect(self.screen, target['color'], rect, 4, border_radius=4)
            pygame.gfxdraw.filled_circle(self.screen, rect.centerx, rect.centery, 5, (*target['color'], 100))

        # Draw blocks
        for block in self.blocks:
            x, y = block['pos']
            rect = pygame.Rect(ox + x * cs + 2, oy + y * cs + 2, cs - 4, cs - 4)
            
            pygame.draw.rect(self.screen, block['color'], rect, 0, border_radius=4)
            
            highlight = tuple(min(255, c + 40) for c in block['color'])
            shadow = tuple(max(0, c - 40) for c in block['color'])
            pygame.draw.line(self.screen, highlight, rect.topleft, rect.topright, 2)
            pygame.draw.line(self.screen, highlight, rect.topleft, rect.bottomleft, 2)
            pygame.draw.line(self.screen, shadow, rect.bottomleft, rect.bottomright, 2)
            pygame.draw.line(self.screen, shadow, rect.topright, rect.bottomright, 2)
            
            if block['on_target']:
                # sfx_block_is_locked_indicator()
                pygame.gfxdraw.filled_circle(self.screen, rect.centerx, rect.centery, 6, (255, 255, 255))
                pygame.gfxdraw.aacircle(self.screen, rect.centerx, rect.centery, 6, (255, 255, 255))

    def _render_ui(self):
        """Renders UI elements like score, moves, and messages."""
        moves_surf = self.font_main.render(f"MOVES: {self.moves_remaining}", True, self.COLOR_TEXT)
        self.screen.blit(moves_surf, (10, 10))
        
        score_surf = self.font_main.render(f"SCORE: {self.score:.1f}", True, self.COLOR_TEXT)
        score_rect = score_surf.get_rect(topright=(630, 10))
        self.screen.blit(score_surf, score_rect)
        
        if self.bonus_text_timer > 0:
            bonus_surf = self.font_bonus.render(self.bonus_text, True, self.COLOR_BONUS)
            bonus_rect = bonus_surf.get_rect(center=(320, 380))
            self.screen.blit(bonus_surf, bonus_rect)
            
        if self.game_over:
            overlay = pygame.Surface((640, 400), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            msg, color = ("LEVEL COMPLETE!", self.COLOR_WIN) if self._check_win_condition() else ("OUT OF MOVES", self.COLOR_LOSS)
            end_surf = self.font_main.render(msg, True, color)
            end_rect = end_surf.get_rect(center=(320, 180))
            self.screen.blit(end_surf, end_rect)
            
            final_score_surf = self.font_main.render(f"FINAL SCORE: {self.score:.1f}", True, self.COLOR_TEXT)
            final_score_rect = final_score_surf.get_rect(center=(320, 220))
            self.screen.blit(final_score_surf, final_score_rect)

    def _get_info(self):
        """Returns a dictionary with auxiliary diagnostic information."""
        return {
            "score": self.score,
            "steps": self.steps,
            "moves_remaining": self.moves_remaining,
            "blocks_on_target": sum(1 for b in self.blocks if b['on_target']),
        }

    def validate_implementation(self):
        """
        Performs self-checks to ensure the environment conforms to the
        Gymnasium API and the design brief's specifications.
        """
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
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

    def close(self):
        """Cleans up Pygame resources."""
        pygame.quit()


if __name__ == '__main__':
    # This block allows you to run the file directly to play the game.
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # Setup Pygame window for human play
    pygame.display.set_caption("Block Pusher")
    screen = pygame.display.set_mode((640, 400))
    clock = pygame.time.Clock()
    
    running = True
    game_over = False
    
    print("\n" + "="*30)
    print(f"GAME: {env.game_description}")
    print(f"CONTROLS: {env.user_guide}")
    print("="*30 + "\n")

    while running:
        action = [0, 0, 0] # Default action: [no-op, space_up, shift_up]
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if not game_over and event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    action[0] = 1
                elif event.key == pygame.K_DOWN:
                    action[0] = 2
                elif event.key == pygame.K_LEFT:
                    action[0] = 3
                elif event.key == pygame.K_RIGHT:
                    action[0] = 4
                elif event.key == pygame.K_r: # Reset key
                    obs, info = env.reset()
                    game_over = False
                    action = [0, 0, 0] # Clear action after reset

        if not game_over and action[0] != 0:
            obs, reward, terminated, truncated, info = env.step(action)
            game_over = terminated
            print(f"Action: {action}, Reward: {reward:.2f}, Score: {info['score']:.2f}, Moves Left: {info['moves_remaining']}")

        # Render the observation from the environment
        frame = env._get_observation()
        frame = np.transpose(frame, (1, 0, 2)) # Transpose back for pygame display
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        clock.tick(30) # Limit frame rate to 30 FPS

    env.close()