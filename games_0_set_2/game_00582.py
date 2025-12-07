
# Generated: 2025-08-27T14:05:48.792375
# Source Brief: brief_00582.md
# Brief Index: 582

        
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
        "Controls: ↑↓←→ to push all blocks in a direction. Get each block to its matching goal."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Push colored blocks to their matching goals in a grid-based puzzle with a limited number of moves."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # --- Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.GRID_W, self.GRID_H = 16, 8
        self.CELL_SIZE = 40
        self.GRID_PIXEL_W = self.GRID_W * self.CELL_SIZE
        self.GRID_PIXEL_H = self.GRID_H * self.CELL_SIZE
        self.GRID_OFFSET_X = (self.WIDTH - self.GRID_PIXEL_W) // 2
        self.GRID_OFFSET_Y = (self.HEIGHT - self.GRID_PIXEL_H) // 2 + 20

        # --- Colors ---
        self.COLOR_BG = (25, 25, 35)
        self.COLOR_GRID = (50, 50, 60)
        self.COLOR_WALL = (10, 10, 15)
        self.COLOR_TEXT = (220, 220, 220)
        self.COLOR_TEXT_SHADOW = (10, 10, 10)
        self.COLOR_OVERLAY = (0, 0, 0, 180)

        self.BLOCK_COLORS = [
            (255, 80, 80),   # Red
            (80, 255, 80),   # Green
            (80, 150, 255),  # Blue
            (255, 255, 80),  # Yellow
            (255, 80, 255),  # Magenta
            (80, 255, 255),  # Cyan
            (255, 160, 80),  # Orange
            (160, 80, 255),  # Purple
        ]
        
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
        self.font_ui = pygame.font.SysFont("Consolas", 20, bold=True)
        self.font_big = pygame.font.SysFont("Consolas", 48, bold=True)
        
        # Etc...        
        
        # Initialize state variables
        self.level = 0
        self.steps = 0
        self.score = 0
        self.moves_left = 0
        self.max_moves = 0
        self.game_over = True # Force level generation on first reset
        self.game_won = False
        self.walls = set()
        self.blocks = []
        self.goals = []
        self.goals_dict = {}
        self.solved_blocks = set()

        self.reset()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # On win, advance level. On loss, retry same level.
        if not self.game_over or self.game_won:
            self.level += 1
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.game_won = False
        
        self._generate_puzzle()
        
        # MUST return exactly this tuple
        return self._get_observation(), self._get_info()

    def _generate_puzzle(self):
        self.walls.clear()
        self.blocks.clear()
        self.goals.clear()
        self.goals_dict.clear()
        self.solved_blocks.clear()

        # Create border walls
        for x in range(-1, self.GRID_W + 1):
            self.walls.add((x, -1))
            self.walls.add((x, self.GRID_H))
        for y in range(self.GRID_H):
            self.walls.add((-1, y))
            self.walls.add((self.GRID_W, y))

        num_blocks = min(len(self.BLOCK_COLORS), 2 + self.level)
        self.max_moves = 40 + self.level * 10
        self.moves_left = self.max_moves

        available_cells = []
        for x in range(self.GRID_W):
            for y in range(self.GRID_H):
                available_cells.append((x, y))
        
        self.np_random.shuffle(available_cells)

        # Place goals and initial blocks on goals
        colors = self.np_random.permutation(self.BLOCK_COLORS)[:num_blocks].tolist()
        for i in range(num_blocks):
            goal_pos = available_cells.pop()
            block_pos = goal_pos
            color = tuple(colors[i])
            
            self.goals.append({'pos': goal_pos, 'color': color, 'id': i})
            self.goals_dict[i] = self.goals[-1]
            self.blocks.append({'pos': block_pos, 'color': color, 'id': i})

        # Scramble the puzzle by applying random moves in reverse
        scramble_steps = num_blocks * 5 + self.level * 5
        for _ in range(scramble_steps):
            rand_dx, rand_dy = self.np_random.choice([-1, 0, 1], size=2, replace=False)
            if rand_dx == 0 and rand_dy == 0: # Ensure there's a move
                rand_dx = self.np_random.choice([-1, 1])

            self._apply_push(rand_dx, rand_dy, is_scrambling=True)
            
        self._update_solved_blocks() # Check if any blocks are solved post-scramble
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # Unpack factorized action
        movement = action[0]  # 0-4: none/up/down/left/right
        
        reward = 0

        if movement > 0:
            self.steps += 1
            self.moves_left -= 1
            reward -= 0.1 # Cost per move attempt

            move_map = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}
            dx, dy = move_map[movement]
            
            # # Placeholder: Play 'push' sound
            self._apply_push(dx, dy)
            
            newly_solved_reward = self._update_solved_blocks()
            reward += newly_solved_reward
            self.score += newly_solved_reward

        terminated = self._check_termination()
        if terminated and not self.game_over: # Only apply terminal reward once
            if self.game_won:
                reward += 100
                self.score += 100
                # # Placeholder: Play 'win' sound
            else: # Out of moves
                reward -= 50
                self.score -= 50
                # # Placeholder: Play 'lose' sound
            self.game_over = True
        
        # MUST return exactly this 5-tuple
        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )

    def _apply_push(self, dx, dy, is_scrambling=False):
        # Sort blocks to push them correctly in sequence.
        # Furthest blocks in the direction of the push move first.
        sorted_blocks = sorted(self.blocks, key=lambda b: b['pos'][0] * -dx + b['pos'][1] * -dy)
        
        occupied_cells = {b['pos'] for b in self.blocks}

        for block in sorted_blocks:
            if not is_scrambling and block['id'] in self.solved_blocks:
                continue

            current_pos = block['pos']
            next_pos = (current_pos[0] + dx, current_pos[1] + dy)

            if next_pos in self.walls or next_pos in occupied_cells:
                continue
            
            block['pos'] = next_pos
            occupied_cells.remove(current_pos)
            occupied_cells.add(next_pos)

    def _update_solved_blocks(self):
        newly_solved_reward = 0
        for block in self.blocks:
            if block['id'] not in self.solved_blocks:
                goal_pos = self.goals_dict[block['id']]['pos']
                if block['pos'] == goal_pos:
                    self.solved_blocks.add(block['id'])
                    newly_solved_reward += 1.0
                    # # Placeholder: Play 'block solved' sound
        return newly_solved_reward

    def _check_termination(self):
        if len(self.solved_blocks) == len(self.blocks):
            self.game_won = True
            return True
        
        if self.moves_left <= 0:
            return True

        if self.steps >= 1000:
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
        for x in range(self.GRID_W + 1):
            start = (self.GRID_OFFSET_X + x * self.CELL_SIZE, self.GRID_OFFSET_Y)
            end = (self.GRID_OFFSET_X + x * self.CELL_SIZE, self.GRID_OFFSET_Y + self.GRID_PIXEL_H)
            pygame.draw.line(self.screen, self.COLOR_GRID, start, end, 1)
        for y in range(self.GRID_H + 1):
            start = (self.GRID_OFFSET_X, self.GRID_OFFSET_Y + y * self.CELL_SIZE)
            end = (self.GRID_OFFSET_X + self.GRID_PIXEL_W, self.GRID_OFFSET_Y + y * self.CELL_SIZE)
            pygame.draw.line(self.screen, self.COLOR_GRID, start, end, 1)
            
        # Draw goals
        for goal in self.goals:
            gx, gy = goal['pos']
            rect = pygame.Rect(
                self.GRID_OFFSET_X + gx * self.CELL_SIZE,
                self.GRID_OFFSET_Y + gy * self.CELL_SIZE,
                self.CELL_SIZE,
                self.CELL_SIZE
            )
            r, g, b = goal['color']
            goal_color = (r // 3, g // 3, b // 3)
            pygame.draw.rect(self.screen, goal_color, rect)
            pygame.gfxdraw.rectangle(self.screen, rect, goal['color'])
            
        # Draw blocks
        for block in self.blocks:
            bx, by = block['pos']
            rect = pygame.Rect(
                self.GRID_OFFSET_X + bx * self.CELL_SIZE,
                self.GRID_OFFSET_Y + by * self.CELL_SIZE,
                self.CELL_SIZE,
                self.CELL_SIZE
            )
            
            highlight = tuple(min(255, c + 40) for c in block['color'])
            shadow = tuple(max(0, c - 40) for c in block['color'])
            
            inner_rect = rect.inflate(-8, -8)
            pygame.draw.rect(self.screen, shadow, rect)
            pygame.draw.rect(self.screen, highlight, rect.move(-2, -2))
            pygame.draw.rect(self.screen, block['color'], inner_rect)
            
            if block['id'] in self.solved_blocks:
                center_x, center_y = rect.center
                points = [(center_x - 10, center_y), (center_x - 2, center_y + 8), (center_x + 10, center_y - 8)]
                pygame.draw.lines(self.screen, (255, 255, 255), False, points, 4)
                pygame.draw.lines(self.screen, (0, 0, 0), False, points, 2)

    def _render_ui(self):
        def draw_text(text, font, color, pos):
            text_surf = font.render(text, True, self.COLOR_TEXT_SHADOW)
            self.screen.blit(text_surf, (pos[0] + 2, pos[1] + 2))
            text_surf = font.render(text, True, color)
            self.screen.blit(text_surf, pos)

        draw_text(f"Level: {self.level}", self.font_ui, self.COLOR_TEXT, (15, 10))
        moves_text = f"Moves: {self.moves_left}"
        moves_surf = self.font_ui.render(moves_text, True, self.COLOR_TEXT)
        draw_text(moves_text, self.font_ui, self.COLOR_TEXT, (self.WIDTH // 2 - moves_surf.get_width() // 2, 10))
        score_text = f"Score: {self.score}"
        score_surf = self.font_ui.render(score_text, True, self.COLOR_TEXT)
        draw_text(score_text, self.font_ui, self.COLOR_TEXT, (self.WIDTH - score_surf.get_width() - 15, 10))

        if self._check_termination():
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill(self.COLOR_OVERLAY)
            self.screen.blit(overlay, (0, 0))
            
            msg, color = ("PUZZLE SOLVED!", (150, 255, 150)) if self.game_won else ("OUT OF MOVES", (255, 150, 150))
                
            text_surf = self.font_big.render(msg, True, color)
            text_rect = text_surf.get_rect(center=(self.WIDTH // 2, self.HEIGHT // 2 - 20))
            self.screen.blit(text_surf, text_rect)
            
            score_text = f"Final Score: {self.score}"
            score_surf = self.font_ui.render(score_text, True, self.COLOR_TEXT)
            score_rect = score_surf.get_rect(center=(self.WIDTH // 2, self.HEIGHT // 2 + 30))
            self.screen.blit(score_surf, score_rect)
    
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "moves_left": self.moves_left,
            "level": self.level,
            "blocks_solved": len(self.solved_blocks),
            "total_blocks": len(self.blocks),
        }

    def close(self):
        pygame.font.quit()
        pygame.quit()

if __name__ == '__main__':
    # This block allows you to play the game directly.
    env = GameEnv()
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Push Block Puzzle")
    clock = pygame.time.Clock()
    
    running = True
    
    print(env.game_description)
    print(env.user_guide)
    
    while running:
        action = np.array([0, 0, 0]) # Default to no-op
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if env._check_termination():
                    if event.key == pygame.K_r:
                        obs, info = env.reset()
                else:
                    if event.key == pygame.K_UP: action[0] = 1
                    elif event.key == pygame.K_DOWN: action[0] = 2
                    elif event.key == pygame.K_LEFT: action[0] = 3
                    elif event.key == pygame.K_RIGHT: action[0] = 4
                
                if event.key == pygame.K_q: running = False

        if action[0] != 0:
            obs, reward, terminated, truncated, info = env.step(action)
            print(f"Action: {action[0]}, Reward: {reward:.2f}, Terminated: {terminated}, Info: {info}")
            if terminated:
                print("Game Over. Press 'R' to restart.")
        
        # Render the current state from the environment
        frame = obs
        frame = np.transpose(frame, (1, 0, 2)) # Transpose back for pygame display
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        clock.tick(30)
        
    env.close()