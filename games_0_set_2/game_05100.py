
# Generated: 2025-08-28T03:57:35.430557
# Source Brief: brief_05100.md
# Brief Index: 5100

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
from collections import deque
import os
import pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Use arrow keys (↑↓←→) to move your character. Push all orange crates onto the green targets before time runs out."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A fast-paced, grid-based puzzle game. Push crates onto target locations within a strict 60-second (60-move) time limit. Plan your moves carefully!"
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Game constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.GRID_COLS, self.GRID_ROWS = 16, 10
        self.CELL_SIZE = self.HEIGHT // self.GRID_ROWS
        self.NUM_CRATES = 6
        self.MAX_STEPS = 60

        # Colors
        self.COLOR_BG = (25, 25, 35)
        self.COLOR_GRID = (40, 40, 50)
        self.COLOR_WALL = (60, 60, 75)
        self.COLOR_PLAYER = (50, 150, 255)
        self.COLOR_PLAYER_ACCENT = (150, 200, 255)
        self.COLOR_CRATE = (255, 150, 50)
        self.COLOR_CRATE_ACCENT = (255, 200, 150)
        self.COLOR_TARGET = (50, 255, 150)
        self.COLOR_TARGET_FILLED = (40, 200, 120)
        self.COLOR_TEXT = (220, 220, 220)

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
        self.font_large = pygame.font.Font(None, 48)
        self.font_small = pygame.font.Font(None, 32)
        
        # Game state attributes (initialized in reset)
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.player_pos = None
        self.crate_pos = []
        self.target_pos = []
        self.wall_pos = set()
        self.particles = []
        self.crate_target_map = {}
        
        # Initialize state variables
        self.reset()
        
        # Run validation check
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initialize all game state
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.particles = []

        # Generate level layout
        self._generate_level()
        
        # Associate crates with targets for reward calculation
        self.crate_target_map = {i: i for i in range(self.NUM_CRATES)}

        return self._get_observation(), self._get_info()

    def _generate_level(self):
        self.wall_pos = set()
        for r in range(self.GRID_ROWS):
            self.wall_pos.add((0, r))
            self.wall_pos.add((self.GRID_COLS - 1, r))
        for c in range(self.GRID_COLS):
            self.wall_pos.add((c, 0))
            self.wall_pos.add((c, self.GRID_ROWS - 1))

        # Get all available empty cells
        available_cells = []
        for c in range(1, self.GRID_COLS - 1):
            for r in range(1, self.GRID_ROWS - 1):
                available_cells.append((c, r))
        
        # Ensure enough cells for player, crates, and targets
        num_entities = 1 + self.NUM_CRATES * 2
        if len(available_cells) < num_entities:
            raise ValueError("Grid not large enough for all entities.")

        # Randomly place player, crates, and targets without overlap
        chosen_indices = self.np_random.choice(len(available_cells), num_entities, replace=False)
        chosen_cells = [available_cells[i] for i in chosen_indices]

        self.player_pos = chosen_cells.pop(0)
        self.crate_pos = [chosen_cells.pop(0) for _ in range(self.NUM_CRATES)]
        self.target_pos = [chosen_cells.pop(0) for _ in range(self.NUM_CRATES)]

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement = action[0]  # 0-4: none/up/down/left/right
        reward = 0
        
        # Map movement action to a delta (dx, dy)
        move_map = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}
        
        if movement in move_map:
            dx, dy = move_map[movement]
            player_next_pos = (self.player_pos[0] + dx, self.player_pos[1] + dy)

            # Check for collisions
            if player_next_pos in self.wall_pos:
                pass # Player hits a wall, no move
            elif player_next_pos in self.crate_pos:
                crate_index = self.crate_pos.index(player_next_pos)
                crate_next_pos = (player_next_pos[0] + dx, player_next_pos[1] + dy)

                if crate_next_pos not in self.wall_pos and crate_next_pos not in self.crate_pos:
                    # Push is valid
                    target_index = self.crate_target_map[crate_index]
                    target = self.target_pos[target_index]
                    
                    old_dist = self._manhattan_distance(self.crate_pos[crate_index], target)
                    
                    # Move crate and player
                    self.crate_pos[crate_index] = crate_next_pos
                    self.player_pos = player_next_pos
                    
                    new_dist = self._manhattan_distance(self.crate_pos[crate_index], target)

                    # Distance-based reward
                    if new_dist < old_dist:
                        reward += 0.1
                    elif new_dist > old_dist:
                        reward -= 0.1
                    
                    # Check if crate landed on any target
                    if crate_next_pos in self.target_pos:
                        reward += 10
                        self._create_particles(crate_next_pos, self.COLOR_TARGET)
                        # sfx: crate_on_target.wav
                else:
                    pass # Crate push is blocked
            else:
                # Move is to an empty square
                self.player_pos = player_next_pos
                # sfx: player_move.wav

        self.steps += 1
        self._update_particles()
        
        terminated = self._check_termination()
        
        if terminated:
            self.game_over = True
            if self._check_win_condition():
                reward += 100 # Win bonus
                # sfx: win_game.wav
            else:
                reward -= 100 # Loss penalty
                # sfx: lose_game.wav

        self.score += reward
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )

    def _manhattan_distance(self, pos1, pos2):
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    def _check_win_condition(self):
        return set(self.crate_pos) == set(self.target_pos)

    def _check_termination(self):
        if self._check_win_condition():
            return True
        if self.steps >= self.MAX_STEPS:
            return True
        return False
    
    def _get_observation(self):
        # Clear screen with background
        self.screen.fill(self.COLOR_BG)
        
        # Render all game elements
        self._render_game()
        
        # Render UI overlay
        self._render_ui()
        
        # Convert to numpy array
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw grid
        for c in range(self.GRID_COLS):
            pygame.draw.line(self.screen, self.COLOR_GRID, (c * self.CELL_SIZE, 0), (c * self.CELL_SIZE, self.HEIGHT))
        for r in range(self.GRID_ROWS):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, r * self.CELL_SIZE), (self.WIDTH, r * self.CELL_SIZE))

        # Draw targets
        for pos in self.target_pos:
            center_x = int((pos[0] + 0.5) * self.CELL_SIZE)
            center_y = int((pos[1] + 0.5) * self.CELL_SIZE)
            radius = int(self.CELL_SIZE * 0.35)
            color = self.COLOR_TARGET_FILLED if pos in self.crate_pos else self.COLOR_TARGET
            pygame.gfxdraw.filled_circle(self.screen, center_x, center_y, radius, color)
            pygame.gfxdraw.aacircle(self.screen, center_x, center_y, radius, color)

        # Draw walls
        for pos in self.wall_pos:
            rect = pygame.Rect(pos[0] * self.CELL_SIZE, pos[1] * self.CELL_SIZE, self.CELL_SIZE, self.CELL_SIZE)
            pygame.draw.rect(self.screen, self.COLOR_WALL, rect)

        # Draw crates
        for pos in self.crate_pos:
            rect = pygame.Rect(pos[0] * self.CELL_SIZE, pos[1] * self.CELL_SIZE, self.CELL_SIZE, self.CELL_SIZE).inflate(-4, -4)
            pygame.draw.rect(self.screen, self.COLOR_CRATE, rect, border_radius=4)
            pygame.draw.rect(self.screen, self.COLOR_CRATE_ACCENT, rect.inflate(-8, -8), border_radius=2)

        # Draw player
        rect = pygame.Rect(self.player_pos[0] * self.CELL_SIZE, self.player_pos[1] * self.CELL_SIZE, self.CELL_SIZE, self.CELL_SIZE).inflate(-4, -4)
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, rect, border_radius=4)
        pygame.draw.rect(self.screen, self.COLOR_PLAYER_ACCENT, rect.inflate(-8, -8), border_radius=2)
        
        # Draw particles
        for p in self.particles:
            pos, vel, life, max_life, color = p
            alpha = int(255 * (life / max_life))
            temp_surf = pygame.Surface((4, 4), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, color + (alpha,), (2, 2), 2)
            self.screen.blit(temp_surf, (int(pos[0]) - 2, int(pos[1]) - 2))

    def _render_ui(self):
        # Time remaining
        time_text = f"Moves: {self.MAX_STEPS - self.steps}"
        time_surf = self.font_large.render(time_text, True, self.COLOR_TEXT)
        self.screen.blit(time_surf, (self.WIDTH - time_surf.get_width() - 20, 10))

        # Crates on targets
        crates_on_target = len(set(self.crate_pos) & set(self.target_pos))
        crates_text = f"Goal: {crates_on_target} / {self.NUM_CRATES}"
        crates_surf = self.font_large.render(crates_text, True, self.COLOR_TEXT)
        self.screen.blit(crates_surf, (20, 10))
        
        # Game over message
        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            if self._check_win_condition():
                msg = "LEVEL COMPLETE!"
                color = self.COLOR_TARGET
            else:
                msg = "TIME UP!"
                color = self.COLOR_CRATE
                
            msg_surf = self.font_large.render(msg, True, color)
            msg_rect = msg_surf.get_rect(center=(self.WIDTH // 2, self.HEIGHT // 2))
            self.screen.blit(msg_surf, msg_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "crates_on_target": len(set(self.crate_pos) & set(self.target_pos)),
        }

    def _create_particles(self, grid_pos, color):
        center_x = (grid_pos[0] + 0.5) * self.CELL_SIZE
        center_y = (grid_pos[1] + 0.5) * self.CELL_SIZE
        for _ in range(20):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1, 4)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            life = random.randint(15, 30)
            self.particles.append([[center_x, center_y], vel, life, life, color])

    def _update_particles(self):
        for i in range(len(self.particles) - 1, -1, -1):
            p = self.particles[i]
            p[0][0] += p[1][0] # pos.x += vel.x
            p[0][1] += p[1][1] # pos.y += vel.y
            p[1][1] += 0.1     # gravity
            p[2] -= 1          # life -= 1
            if p[2] <= 0:
                self.particles.pop(i)

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
    # This block allows you to play the game manually for testing
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # Setup Pygame window for human play
    pygame.display.set_caption("Sokoban Rush")
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    
    terminated = False
    
    # Map Pygame keys to action space
    key_to_action = {
        pygame.K_UP: 1,
        pygame.K_DOWN: 2,
        pygame.K_LEFT: 3,
        pygame.K_RIGHT: 4,
    }

    print(GameEnv.user_guide)
    
    running = True
    while running:
        action = np.array([0, 0, 0]) # Default no-op
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r: # Reset on 'r'
                    obs, info = env.reset()
                    terminated = False
                    print("--- Game Reset ---")
                elif event.key in key_to_action and not terminated:
                    action[0] = key_to_action[event.key]
                    obs, reward, terminated, _, info = env.step(action)
                    print(f"Step: {info['steps']}, Reward: {reward:.2f}, Score: {info['score']:.2f}, Terminated: {terminated}")

        # Render the observation from the environment
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
    env.close()