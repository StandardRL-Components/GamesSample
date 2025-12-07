
# Generated: 2025-08-28T03:33:24.109645
# Source Brief: brief_04962.md
# Brief Index: 4962

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
from collections import defaultdict
import os
import pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Arrow keys to push all blocks. Solve the puzzle before you run out of moves."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A minimalist puzzle game. Push colored blocks onto their matching targets. Plan your moves carefully!"
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Game constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.GRID_WIDTH, self.GRID_HEIGHT = 12, 8
        self.CELL_SIZE = 40
        self.GRID_X_OFFSET = (self.WIDTH - self.GRID_WIDTH * self.CELL_SIZE) // 2
        self.GRID_Y_OFFSET = (self.HEIGHT - self.GRID_HEIGHT * self.CELL_SIZE) // 2
        self.MAX_STEPS = 1000

        # Colors
        self.COLOR_BG = (44, 62, 80) # #2c3e50
        self.COLOR_GRID = (52, 73, 94) # #34495e
        self.COLOR_TEXT = (236, 240, 241) # #ecf0f1
        self.COLOR_OVERLAY = (44, 62, 80, 200)
        self.BLOCK_COLORS = [
            (231, 76, 60),   # #e74c3c (Red)
            (52, 152, 219),  # #3498db (Blue)
            (46, 204, 113),  # #2ecc71 (Green)
            (241, 196, 15),   # #f1c40f (Yellow)
            (155, 89, 182),  # #9b59b6 (Purple)
            (26, 188, 156),  # #1abc9c (Turquoise)
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
        self.font_main = pygame.font.Font(None, 36)
        self.font_large = pygame.font.Font(None, 72)
        
        # Game state that persists across resets
        self.current_level = 1
        
        # Initialize state variables (will be properly set in reset)
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.moves_left = 0
        self.blocks = []
        self.targets = []
        self.particles = []

        # Initialize state variables
        self.reset()

        # Validate implementation
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        if options and 'level' in options:
            self.current_level = options['level']

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win_state = False
        self.particles = []
        
        self._generate_level()
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        reward = 0
        terminated = False

        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # Unpack factorized action
        movement = action[0]  # 0-4: none/up/down/left/right
        
        self.steps += 1
        
        if movement > 0:
            self.moves_left -= 1
            reward -= 0.1 # Cost for making a move
            
            # Store pre-move matched state
            pre_move_matches = {i for i, b in enumerate(self.blocks) if b['locked']}
            
            # Apply push logic
            self._apply_push(movement)
            # SFX: whoosh.wav
            
            # Check for new matches and update rewards
            for i, block in enumerate(self.blocks):
                if not block['locked']:
                    for target in self.targets:
                        if block['pos'] == target['pos'] and block['color_idx'] == target['color_idx']:
                            block['locked'] = True
                            if i not in pre_move_matches:
                                reward += 0.5 # Reward for a new match
                                self._spawn_particles(block['pos'], block['color_idx'])
                                # SFX: match.wav
                            break
        
        self.score += reward
        terminated, term_reward = self._check_termination()
        self.score += term_reward
        reward += term_reward
        
        if terminated:
            self.game_over = True
            if self.win_state:
                self.current_level += 1 # Progress to next level on next reset
                # SFX: win_jingle.wav
            else:
                # SFX: lose_sound.wav
                pass
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )
    
    def _generate_level(self):
        # Difficulty scaling
        num_targets = min(3 + (self.current_level - 1), len(self.BLOCK_COLORS))
        num_extra_blocks = 2 + (self.current_level - 1)
        num_blocks = num_targets + num_extra_blocks
        self.moves_left = 20 + 5 * (self.current_level - 1)
        
        # Get all possible grid positions
        all_pos = [(x, y) for x in range(self.GRID_WIDTH) for y in range(self.GRID_HEIGHT)]
        self.np_random.shuffle(all_pos)
        
        # Place targets
        self.targets = []
        for i in range(num_targets):
            pos = all_pos.pop()
            self.targets.append({'pos': pos, 'color_idx': i})
            
        # Place blocks in a solved state
        self.blocks = []
        occupied_pos = set()
        for i in range(num_targets):
            pos = self.targets[i]['pos']
            self.blocks.append({'pos': pos, 'color_idx': i, 'locked': False})
            occupied_pos.add(pos)
        
        # Place extra blocks
        for i in range(num_extra_blocks):
            pos = all_pos.pop()
            color_idx = self.np_random.integers(num_targets, len(self.BLOCK_COLORS))
            self.blocks.append({'pos': pos, 'color_idx': color_idx, 'locked': False})
            occupied_pos.add(pos)

        # Shuffle the puzzle by applying reverse moves (guarantees solvability)
        shuffle_moves = self.current_level * 5 + 10
        for _ in range(shuffle_moves):
            # A "reverse" push is just a normal push
            move = self.np_random.integers(1, 5)
            # Temporarily unlock all blocks to shuffle them
            for b in self.blocks: b['locked'] = False
            self._apply_push(move)

        # Reset locked state after shuffling
        for block in self.blocks:
            block['locked'] = False
            for target in self.targets:
                if block['pos'] == target['pos'] and block['color_idx'] == target['color_idx']:
                    block['locked'] = True
                    break

    def _apply_push(self, movement):
        # 1=up, 2=down, 3=left, 4=right
        direction_map = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}
        dx, dy = direction_map[movement]

        # Sort blocks to handle chain reactions correctly
        sort_key = lambda b: b['pos'][0]
        reverse = (dx > 0 or dy > 0)
        if abs(dy) > 0:
            sort_key = lambda b: b['pos'][1]
        
        sorted_blocks = sorted([b for b in self.blocks if not b['locked']], key=sort_key, reverse=reverse)
        
        # All occupied cells act as walls during a push
        occupied_cells = {b['pos'] for b in self.blocks}

        for block in sorted_blocks:
            current_pos = block['pos']
            occupied_cells.remove(current_pos)
            
            new_pos = list(current_pos)
            while True:
                next_pos = (new_pos[0] + dx, new_pos[1] + dy)
                if not (0 <= next_pos[0] < self.GRID_WIDTH and 0 <= next_pos[1] < self.GRID_HEIGHT):
                    break # Hit wall
                if next_pos in occupied_cells:
                    break # Hit another block
                
                new_pos[0], new_pos[1] = next_pos
            
            block['pos'] = tuple(new_pos)
            occupied_cells.add(block['pos'])

    def _check_termination(self):
        # Win condition: all targets are covered by a correct, locked block
        matched_targets = 0
        for target in self.targets:
            for block in self.blocks:
                if block['pos'] == target['pos'] and block['color_idx'] == target['color_idx']:
                    matched_targets += 1
                    break
        
        if matched_targets == len(self.targets):
            self.win_state = True
            return True, 100.0 # Win
        
        # Lose condition: out of moves
        if self.moves_left <= 0:
            self.win_state = False
            return True, -50.0 # Lose
            
        # Max steps
        if self.steps >= self.MAX_STEPS:
            self.win_state = False
            return True, 0.0 # Terminated by steps
            
        return False, 0.0

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._update_and_render_particles()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _render_game(self):
        # Draw grid lines
        for x in range(self.GRID_WIDTH + 1):
            px = self.GRID_X_OFFSET + x * self.CELL_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (px, self.GRID_Y_OFFSET), (px, self.GRID_Y_OFFSET + self.GRID_HEIGHT * self.CELL_SIZE), 1)
        for y in range(self.GRID_HEIGHT + 1):
            py = self.GRID_Y_OFFSET + y * self.CELL_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (self.GRID_X_OFFSET, py), (self.GRID_X_OFFSET + self.GRID_WIDTH * self.CELL_SIZE, py), 1)

        # Draw targets
        for target in self.targets:
            color = self.BLOCK_COLORS[target['color_idx']]
            darker_color = tuple(c * 0.4 for c in color)
            rect = pygame.Rect(
                self.GRID_X_OFFSET + target['pos'][0] * self.CELL_SIZE,
                self.GRID_Y_OFFSET + target['pos'][1] * self.CELL_SIZE,
                self.CELL_SIZE, self.CELL_SIZE
            )
            pygame.draw.rect(self.screen, darker_color, rect)

        # Draw blocks
        for block in self.blocks:
            color = self.BLOCK_COLORS[block['color_idx']]
            rect = pygame.Rect(
                self.GRID_X_OFFSET + block['pos'][0] * self.CELL_SIZE + 4,
                self.GRID_Y_OFFSET + block['pos'][1] * self.CELL_SIZE + 4,
                self.CELL_SIZE - 8, self.CELL_SIZE - 8
            )
            pygame.draw.rect(self.screen, color, rect, border_radius=4)
            if block['locked']:
                # Draw a checkmark/glow to indicate locked status
                pygame.draw.rect(self.screen, self.COLOR_TEXT, rect, 2, border_radius=4)


    def _render_ui(self):
        # Moves left
        moves_text = self.font_main.render(f"Moves: {self.moves_left}", True, self.COLOR_TEXT)
        self.screen.blit(moves_text, (20, 15))
        
        # Level
        level_text = self.font_main.render(f"Level: {self.current_level}", True, self.COLOR_TEXT)
        self.screen.blit(level_text, (self.WIDTH - level_text.get_width() - 20, 15))

        # Score
        score_text = self.font_main.render(f"Score: {self.score:.1f}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (20, 50))

        # Game Over Screen
        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill(self.COLOR_OVERLAY)
            self.screen.blit(overlay, (0, 0))
            
            msg = "LEVEL COMPLETE!" if self.win_state else "OUT OF MOVES"
            end_text = self.font_large.render(msg, True, self.COLOR_TEXT)
            text_rect = end_text.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2))
            self.screen.blit(end_text, text_rect)

    def _spawn_particles(self, grid_pos, color_idx):
        px = self.GRID_X_OFFSET + (grid_pos[0] + 0.5) * self.CELL_SIZE
        py = self.GRID_Y_OFFSET + (grid_pos[1] + 0.5) * self.CELL_SIZE
        color = self.BLOCK_COLORS[color_idx]
        
        for _ in range(20):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            self.particles.append({
                'x': px, 'y': py,
                'vx': math.cos(angle) * speed,
                'vy': math.sin(angle) * speed,
                'life': self.np_random.integers(15, 30),
                'color': color,
                'size': self.np_random.uniform(2, 5)
            })
    
    def _update_and_render_particles(self):
        active_particles = []
        for p in self.particles:
            p['x'] += p['vx']
            p['y'] += p['vy']
            p['life'] -= 1
            p['size'] -= 0.1
            if p['life'] > 0 and p['size'] > 0:
                pygame.draw.circle(self.screen, p['color'], (int(p['x']), int(p['y'])), int(p['size']))
                active_particles.append(p)
        self.particles = active_particles

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "moves_left": self.moves_left,
            "level": self.current_level,
            "win": self.win_state,
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

if __name__ == '__main__':
    # This block allows you to play the game directly
    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    # Use a persistent pygame screen for human play
    pygame.display.set_caption("Block Pusher")
    human_screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    
    action = env.action_space.sample()
    action[0] = 0 # Start with a no-op
    
    print(env.user_guide)

    while not done:
        # Human input
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    action[0] = 1
                elif event.key == pygame.K_DOWN:
                    action[0] = 2
                elif event.key == pygame.K_LEFT:
                    action[0] = 3
                elif event.key == pygame.K_RIGHT:
                    action[0] = 4
                elif event.key == pygame.K_r: # Press 'R' to reset
                    obs, info = env.reset()
                    action[0] = 0
                else:
                    action[0] = 0 # No-op for other keys
                
                # In turn-based, we step immediately on key press
                if action[0] != 0:
                    obs, reward, terminated, truncated, info = env.step(action)
                    print(f"Action: {action[0]}, Reward: {reward:.2f}, Terminated: {terminated}, Info: {info}")
                    if terminated:
                        print("Game Over! Press 'R' to play again.")
                
                # Reset action to no-op for next frame
                action[0] = 0
        
        # Rendering for human display
        # The observation is already a rendered frame
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        human_screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        env.clock.tick(30) # Limit FPS for human play
        
    pygame.quit()