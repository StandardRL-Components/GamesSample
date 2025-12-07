
# Generated: 2025-08-28T00:28:50.004313
# Source Brief: brief_03804.md
# Brief Index: 3804

        
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
        "Controls: ←→ to move and push boxes."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Push boxes onto target locations as quickly as possible in this side-view puzzle game."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Game constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.GRID_SIZE = 40
        self.GRID_W = self.WIDTH // self.GRID_SIZE
        self.GRID_H = self.HEIGHT // self.GRID_SIZE
        self.MAX_LEVELS = 3
        self.MAX_STEPS_PER_LEVEL = 150

        # Colors
        self.COLOR_BG = (25, 28, 32)
        self.COLOR_GRID = (40, 44, 52)
        self.COLOR_PLAYER = (229, 90, 85) # Bright Red
        self.COLOR_BOX = (198, 120, 221) # Bright Purple
        self.COLOR_TARGET = (80, 150, 90) # Muted Green
        self.COLOR_TARGET_ACTIVE = (152, 195, 121) # Bright Green
        self.COLOR_SHADOW = (18, 20, 23)
        self.COLOR_UI_TEXT = (200, 205, 215)
        
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
        self.font_large = pygame.font.Font(None, 36)
        self.font_small = pygame.font.Font(None, 28)
        
        # Level definitions: [player_start, [box_starts], [target_locs]]
        self.levels = [
            {
                "player": [2, 8],
                "boxes": [[5, 8], [8, 8], [11, 8]],
                "targets": [[7, 8], [9, 8], [13, 8]]
            },
            {
                "player": [1, 8],
                "boxes": [[4, 8], [7, 8], [8, 6]],
                "targets": [[10, 8], [11, 8], [12, 6]]
            },
            {
                "player": [7, 8],
                "boxes": [[4, 8], [7, 6], [10, 8]],
                "targets": [[2, 8], [7, 4], [12, 8]]
            }
        ]
        
        # Initialize state variables
        self.player_pos = [0, 0]
        self.box_positions = []
        self.target_positions = []
        self.current_level = 0
        self.steps_in_level = 0
        self.particles = []
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.last_covered_count = 0

        self.reset()
        self.validate_implementation()
    
    def _load_level(self, level_index):
        if level_index >= len(self.levels):
            self.game_over = True
            return

        level_data = self.levels[level_index]
        self.player_pos = list(level_data["player"])
        self.box_positions = [list(pos) for pos in level_data["boxes"]]
        self.target_positions = [list(pos) for pos in level_data["targets"]]
        
        self.steps_in_level = 0
        self.particles = []
        self.last_covered_count = self._get_covered_targets_count()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.current_level = 0
        self._load_level(self.current_level)
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # Unpack factorized action
        movement = action[0]  # 0-4: none/up/down/left/right
        
        self.steps += 1
        self.steps_in_level += 1
        
        reward = -0.01  # Small penalty for taking a step

        # --- Game Logic ---
        old_box_positions = [list(p) for p in self.box_positions]
        old_distances = self._calculate_total_distance()

        if movement in [3, 4]: # 3=left, 4=right
            self._handle_movement(movement)
        
        # --- Reward Calculation ---
        reward += self._calculate_reward(old_box_positions, old_distances)
        self.score += reward

        # --- Termination Check ---
        terminated = self._check_termination()
        
        if self.game_over: # Game over can be set by _check_termination
            terminated = True
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )

    def _handle_movement(self, movement):
        # Sfx: Player step sound
        dx = -1 if movement == 3 else 1
        
        player_x, player_y = self.player_pos
        target_x, target_y = player_x + dx, player_y

        # Check wall collision
        if not (0 <= target_x < self.GRID_W):
            return

        box_idx = self._get_box_at([target_x, target_y])
        
        # If pushing a box
        if box_idx is not None:
            box_target_x, box_target_y = target_x + dx, target_y
            
            # Check if box can be pushed
            if not (0 <= box_target_x < self.GRID_W):
                return # Box hits wall
            if self._get_box_at([box_target_x, box_target_y]) is not None:
                return # Box hits another box
            
            # Move box and player
            # Sfx: Box push sound
            self.box_positions[box_idx][0] = box_target_x
            self.player_pos[0] = target_x
        else:
            # Move player into empty space
            self.player_pos[0] = target_x

    def _calculate_reward(self, old_box_positions, old_distances):
        reward = 0
        
        # 1. Reward for covering/uncovering targets
        current_covered_count = self._get_covered_targets_count()
        if current_covered_count > self.last_covered_count:
            # Sfx: Target covered success sound
            reward += 10 * (current_covered_count - self.last_covered_count)
            # Find which box just covered a target and spawn particles
            for i, new_pos in enumerate(self.box_positions):
                if new_pos != old_box_positions[i] and new_pos in self.target_positions:
                    self._spawn_particles(new_pos)
        elif current_covered_count < self.last_covered_count:
            reward -= 10 * (self.last_covered_count - current_covered_count)
        self.last_covered_count = current_covered_count

        # 2. Reward for moving boxes closer to targets
        new_distances = self._calculate_total_distance()
        distance_delta = old_distances - new_distances
        reward += distance_delta * 0.1

        return reward

    def _check_termination(self):
        # Win condition: all targets covered
        if self._get_covered_targets_count() == len(self.target_positions):
            self.score += 100 # Use score, not reward, for terminal bonus
            self.current_level += 1
            if self.current_level >= self.MAX_LEVELS:
                self.game_over = True
                return True
            else:
                self._load_level(self.current_level)
                return False # Not terminal, just advancing level

        # Loss condition: time out
        if self.steps_in_level >= self.MAX_STEPS_PER_LEVEL:
            self.score -= 100
            self.game_over = True
            return True

        # Loss condition: stuck state
        if self._is_stuck():
            self.score -= 100
            self.game_over = True
            return True
        
        return False

    def _get_observation(self):
        # Clear screen with background
        self.screen.fill(self.COLOR_BG)
        
        self._update_and_draw_particles()
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw targets
        for tx, ty in self.target_positions:
            is_covered = self._get_box_at([tx, ty]) is not None
            color = self.COLOR_TARGET_ACTIVE if is_covered else self.COLOR_TARGET
            rect = pygame.Rect(tx * self.GRID_SIZE, ty * self.GRID_SIZE, self.GRID_SIZE, self.GRID_SIZE)
            pygame.draw.rect(self.screen, color, rect, border_radius=4)
    
        # Draw boxes
        for bx, by in self.box_positions:
            self._draw_entity(self.COLOR_BOX, [bx, by])

        # Draw player
        self._draw_entity(self.COLOR_PLAYER, self.player_pos)
    
    def _draw_entity(self, color, pos):
        x, y = pos
        shadow_offset = self.GRID_SIZE // 10
        rect = pygame.Rect(x * self.GRID_SIZE, y * self.GRID_SIZE, self.GRID_SIZE, self.GRID_SIZE)
        shadow_rect = rect.copy()
        shadow_rect.move_ip(shadow_offset, shadow_offset)
        
        pygame.draw.rect(self.screen, self.COLOR_SHADOW, shadow_rect, border_radius=8)
        pygame.draw.rect(self.screen, color, rect, border_radius=8)

    def _update_and_draw_particles(self):
        active_particles = []
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['vel'][1] += 0.1 # Gravity
            p['lifetime'] -= 1
            if p['lifetime'] > 0:
                active_particles.append(p)
                radius = int(p['radius'] * (p['lifetime'] / p['max_lifetime']))
                if radius > 0:
                    pos_int = (int(p['pos'][0]), int(p['pos'][1]))
                    pygame.gfxdraw.filled_circle(self.screen, pos_int[0], pos_int[1], radius, p['color'])
        self.particles = active_particles

    def _render_ui(self):
        # Level text
        level_text = self.font_large.render(f"Level: {self.current_level + 1}/{self.MAX_LEVELS}", True, self.COLOR_UI_TEXT)
        self.screen.blit(level_text, (15, 10))

        # Steps remaining text
        steps_left = self.MAX_STEPS_PER_LEVEL - self.steps_in_level
        steps_text = self.font_large.render(f"Moves: {max(0, steps_left)}", True, self.COLOR_UI_TEXT)
        self.screen.blit(steps_text, (self.WIDTH - steps_text.get_width() - 15, 10))

        # Score text
        score_text = self.font_small.render(f"Score: {self.score}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (15, 45))

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "level": self.current_level + 1,
            "moves_left": self.MAX_STEPS_PER_LEVEL - self.steps_in_level,
        }

    # --- Helper Functions ---
    def _get_box_at(self, pos):
        try:
            return self.box_positions.index(list(pos))
        except ValueError:
            return None

    def _get_covered_targets_count(self):
        count = 0
        for target_pos in self.target_positions:
            if self._get_box_at(target_pos) is not None:
                count += 1
        return count

    def _calculate_total_distance(self):
        total_dist = 0
        uncovered_targets = [t for t in self.target_positions if self._get_box_at(t) is None]
        if not uncovered_targets:
            return 0
            
        for box_pos in self.box_positions:
            if box_pos not in self.target_positions:
                min_dist = min([abs(box_pos[0] - t[0]) + abs(box_pos[1] - t[1]) for t in uncovered_targets])
                total_dist += min_dist
        return total_dist

    def _is_stuck(self):
        uncovered_targets = [t for t in self.target_positions if self._get_box_at(t) is None]
        if not uncovered_targets:
            return False

        for i, box_pos in enumerate(self.box_positions):
            if box_pos in self.target_positions:
                continue # Box is already on a target, not relevant for stuck check

            x, y = box_pos
            
            # Check left push possibility
            can_push_left = (x > 0) and \
                            (self._get_box_at([x-1, y]) is None) and \
                            (x+1 < self.GRID_W) and \
                            (self._get_box_at([x+1, y]) is None or self.player_pos == [x+1, y])

            # Check right push possibility
            can_push_right = (x < self.GRID_W - 1) and \
                             (self._get_box_at([x+1, y]) is None) and \
                             (x-1 >= 0) and \
                             (self._get_box_at([x-1, y]) is None or self.player_pos == [x-1, y])

            if can_push_left or can_push_right:
                return False # At least one box can be moved

        return True # All non-target boxes are stuck

    def _spawn_particles(self, grid_pos):
        px = (grid_pos[0] + 0.5) * self.GRID_SIZE
        py = (grid_pos[1] + 0.5) * self.GRID_SIZE
        for _ in range(20):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1, 4)
            lifetime = random.randint(20, 40)
            self.particles.append({
                'pos': [px, py],
                'vel': [math.cos(angle) * speed, math.sin(angle) * speed - 2],
                'radius': random.randint(3, 6),
                'lifetime': lifetime,
                'max_lifetime': lifetime,
                'color': random.choice([self.COLOR_TARGET_ACTIVE, (180, 220, 150)])
            })

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

# Example of how to run the environment
if __name__ == '__main__':
    # For human play
    import pygame
    
    env = GameEnv()
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Box Pusher")
    
    terminated = False
    clock = pygame.time.Clock()
    
    print("\n" + "="*30)
    print("Box Pusher - Manual Control")
    print(env.game_description)
    print(env.user_guide)
    print("="*30 + "\n")

    while not terminated:
        action = np.array([0, 0, 0]) # Default no-op
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset() # Allow resetting mid-game
                print("--- Game Reset ---")

        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            action[0] = 3
        elif keys[pygame.K_RIGHT]:
            action[0] = 4
        
        if action[0] != 0: # Only step if an action is taken
            obs, reward, terminated, truncated, info = env.step(action)
            print(f"Step: {info['steps']}, Level: {info['level']}, Reward: {reward:.2f}, Score: {info['score']}, Terminated: {terminated}")
            if terminated:
                print("--- GAME OVER ---")
                print(f"Final Score: {info['score']}")

        # Render the observation to the display
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(10) # Limit frame rate for human play

    pygame.quit()