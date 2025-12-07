
# Generated: 2025-08-27T21:01:39.231240
# Source Brief: brief_02650.md
# Brief Index: 2650

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
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
        "Controls: ↑/↓ to select a box, ←/→ to push the selected box. "
        "Each action costs one move."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Push all the brown boxes onto the green target zones. "
        "You have a limited number of moves to solve the puzzle."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    NUM_BOXES = 3
    NUM_PLATFORMS = 3
    MAX_MOVES = 20
    
    GROUND_Y = SCREEN_HEIGHT - 40
    BOX_SIZE = 40
    PLATFORM_HEIGHT = 15
    PLATFORM_WIDTH_MIN = 60
    PLATFORM_WIDTH_MAX = 120
    PUSH_STRENGTH = 40 # How many pixels a push moves a box
    GRAVITY = 10 # How many pixels a box falls per physics step

    # --- Colors ---
    COLOR_BG = (240, 245, 250)
    COLOR_PLATFORM = (180, 190, 200)
    COLOR_TARGET = (140, 220, 150)
    COLOR_BOX = (160, 110, 80)
    COLOR_BOX_ON_TARGET = (100, 160, 100)
    COLOR_SELECTOR = (60, 120, 240)
    COLOR_TEXT = (50, 50, 70)
    COLOR_TEXT_SUCCESS = (0, 150, 0)
    COLOR_TEXT_FAIL = (200, 0, 0)

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        
        self.font_main = pygame.font.SysFont("sans-serif", 24)
        self.font_game_over = pygame.font.SysFont("sans-serif", 64, bold=True)
        
        self.active_box_index = 0
        self.boxes = []
        self.platforms = []
        self.targets = []
        self.boxes_on_target = []
        
        self.steps = 0
        self.score = 0
        self.moves_left = 0
        self.game_over = False

        # self.reset() is called here to initialize state
        # but the official return values are handled by the user's first call
        # to env.reset()
        self.reset()
        
        # self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.moves_left = self.MAX_MOVES
        self.active_box_index = 0
        
        self._generate_level()
        self._update_boxes_on_target_status()
        
        return self._get_observation(), self._get_info()

    def _generate_level(self):
        """Procedurally generates a new puzzle layout."""
        self.platforms = []
        self.targets = []
        self.boxes = []

        # Generate platforms, ensuring they don't overlap
        min_y = self.GROUND_Y - 3 * self.BOX_SIZE # Min height to allow stacking
        max_y = self.GROUND_Y - self.PLATFORM_HEIGHT
        
        for _ in range(self.NUM_PLATFORMS):
            placed = False
            for _ in range(100): # Max 100 attempts to prevent infinite loop
                width = self.np_random.integers(self.PLATFORM_WIDTH_MIN, self.PLATFORM_WIDTH_MAX + 1)
                x = self.np_random.integers(0, self.SCREEN_WIDTH - width)
                y = self.np_random.integers(min_y, max_y)
                new_platform = pygame.Rect(x, y, width, self.PLATFORM_HEIGHT)
                
                # Ensure it doesn't overlap with existing platforms
                if not any(new_platform.colliderect(p) for p in self.platforms):
                    self.platforms.append(new_platform)
                    # Target is a slightly smaller rect on top of the platform
                    target = pygame.Rect(new_platform.x + 5, new_platform.y - 5, new_platform.width - 10, 5)
                    self.targets.append(target)
                    placed = True
                    break
            if not placed:
                # Fallback if placement fails, though unlikely with these parameters
                self.platforms.append(pygame.Rect(100, 300, 100, self.PLATFORM_HEIGHT))
                self.targets.append(pygame.Rect(105, 295, 90, 5))


        # Generate boxes on the ground, ensuring they don't overlap
        for i in range(self.NUM_BOXES):
            placed = False
            for _ in range(100):
                x = self.np_random.integers(0, self.SCREEN_WIDTH - self.BOX_SIZE)
                new_box = pygame.Rect(x, self.GROUND_Y - self.BOX_SIZE, self.BOX_SIZE, self.BOX_SIZE)
                if not any(new_box.colliderect(b) for b in self.boxes):
                    self.boxes.append(new_box)
                    placed = True
                    break
            if not placed:
                self.boxes.append(pygame.Rect(i * (self.BOX_SIZE + 10), self.GROUND_Y - self.BOX_SIZE, self.BOX_SIZE, self.BOX_SIZE))
        
        self._apply_physics()


    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement = action[0]
        reward = 0
        
        # Every action consumes a move
        self.moves_left -= 1
        self.steps += 1
        
        # Store state for reward calculation
        active_box = self.boxes[self.active_box_index]
        old_box_pos = active_box.copy()
        
        # --- Action Handling ---
        if movement == 1:  # Up: Select next box
            self.active_box_index = (self.active_box_index + 1) % self.NUM_BOXES
            # sound: selection_tick.wav
        elif movement == 2:  # Down: Select previous box
            self.active_box_index = (self.active_box_index - 1 + self.NUM_BOXES) % self.NUM_BOXES
            # sound: selection_tick.wav
        elif movement == 3:  # Left: Push selected box
            active_box.x -= self.PUSH_STRENGTH
            # sound: push.wav
        elif movement == 4:  # Right: Push selected box
            active_box.x += self.PUSH_STRENGTH
            # sound: push.wav
            
        # --- Physics and Collision ---
        self._apply_physics()
        
        # --- Reward Calculation ---
        if movement in [3, 4]: # Only calculate distance reward for pushes
             reward += self._calculate_distance_reward(old_box_pos, self.boxes[self.active_box_index])

        # Check for new boxes on targets
        old_boxes_on_target_count = sum(self.boxes_on_target)
        self._update_boxes_on_target_status()
        new_boxes_on_target_count = sum(self.boxes_on_target)
        
        if new_boxes_on_target_count > old_boxes_on_target_count:
            # sound: success_chime.wav
            placed_reward = (new_boxes_on_target_count - old_boxes_on_target_count) * 5
            reward += placed_reward
            self.score += placed_reward

        # --- Termination Check ---
        win_condition = all(self.boxes_on_target)
        loss_condition = self.moves_left <= 0
        terminated = win_condition or loss_condition
        
        if terminated:
            self.game_over = True
            if win_condition:
                # sound: level_complete.wav
                reward += 50
                self.score += 50
            elif loss_condition and not win_condition:
                # sound: failure_buzzer.wav
                reward -= 50
                self.score -= 50
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _apply_physics(self):
        """Handle collisions and gravity for all boxes."""
        # Clamp boxes to screen bounds horizontally
        for box in self.boxes:
            box.left = max(0, box.left)
            box.right = min(self.SCREEN_WIDTH, box.right)

        # Resolve horizontal collisions between boxes iteratively
        for _ in range(self.NUM_BOXES):
            for i in range(self.NUM_BOXES):
                for j in range(i + 1, self.NUM_BOXES):
                    box1 = self.boxes[i]
                    box2 = self.boxes[j]
                    if box1.colliderect(box2):
                        dx = (box1.centerx - box2.centerx) / 2.0
                        dy = (box1.centery - box2.centery) / 2.0
                        
                        # Separate primarily along the shorter axis of overlap
                        if abs((box1.width / 2 + box2.width / 2) - abs(dx)) < abs((box1.height / 2 + box2.height / 2) - abs(dy)):
                            overlap = (box1.width / 2 + box2.width / 2) - abs(dx)
                            if dx > 0:
                                box1.x += overlap / 2
                                box2.x -= overlap / 2
                            else:
                                box1.x -= overlap / 2
                                box2.x += overlap / 2


        # Apply gravity until all boxes are stable
        stable = False
        for _ in range(20): # Max iterations to prevent infinite loops
            if stable: break
            stable = True
            for i, box in enumerate(self.boxes):
                box.bottom = min(box.bottom, self.GROUND_Y) # Cannot fall through ground
                is_supported = box.bottom >= self.GROUND_Y
                
                support_y = self.GROUND_Y
                
                # Check for platform support
                for plat in self.platforms:
                    if box.colliderect(plat) and box.centerx > plat.left and box.centerx < plat.right:
                        support_y = min(support_y, plat.top)
                
                # Check for other box support
                for j, other_box in enumerate(self.boxes):
                    if i == j: continue
                    if box.colliderect(other_box) and box.bottom <= other_box.top:
                        support_y = min(support_y, other_box.top)
                
                # If box is above its highest support, it falls
                if box.bottom < support_y:
                    box.bottom = min(support_y, box.bottom + self.GRAVITY)
                    stable = False

    def _update_boxes_on_target_status(self):
        """Checks which boxes are correctly placed on any target."""
        self.boxes_on_target = [False] * self.NUM_BOXES
        claimed_targets = []
        
        for i, box in enumerate(self.boxes):
            for j, target in enumerate(self.targets):
                if j in claimed_targets:
                    continue
                
                platform = self.platforms[j]
                is_horizontally_aligned = box.centerx > platform.left and box.centerx < platform.right
                is_vertically_aligned = abs(box.bottom - platform.top) < 5
                
                if is_horizontally_aligned and is_vertically_aligned:
                    self.boxes_on_target[i] = True
                    claimed_targets.append(j)
                    break

    def _calculate_distance_reward(self, old_box_rect, new_box_rect):
        """Reward for moving a box closer to an unoccupied target."""
        unclaimed_target_indices = [i for i, on_target in enumerate(self.boxes_on_target) if not on_target]
        
        # Consider only targets not already occupied by OTHER boxes
        current_box_was_on_target = False
        for i, box in enumerate(self.boxes):
            if box == new_box_rect:
                if self.boxes_on_target[i]:
                    current_box_was_on_target = True
                break

        unoccupied_targets = [self.targets[i] for i in unclaimed_target_indices]
        if current_box_was_on_target and new_box_rect != old_box_rect:
            unoccupied_targets.append(self.targets[self.boxes.index(new_box_rect)])


        if not unoccupied_targets:
            return 0

        def min_dist(box_rect, targets):
            box_center = box_rect.center
            return min(
                math.hypot(box_center[0] - t.centerx, box_center[1] - t.centery)
                for t in targets
            )

        old_min_dist = min_dist(old_box_rect, unoccupied_targets)
        new_min_dist = min_dist(new_box_rect, unoccupied_targets)

        if new_min_dist < old_min_dist:
            return 0.1
        elif new_min_dist > old_min_dist:
            return -0.1
        return 0

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw platforms
        for plat in self.platforms:
            pygame.draw.rect(self.screen, self.COLOR_PLATFORM, plat)
        
        # Draw targets
        for i, target in enumerate(self.targets):
             # Draw target on top of its platform
            target_rect = pygame.Rect(self.platforms[i].x+5, self.platforms[i].y-5, self.platforms[i].width-10, 5)
            pygame.draw.rect(self.screen, self.COLOR_TARGET, target_rect)

        # Draw boxes
        for i, box in enumerate(self.boxes):
            color = self.COLOR_BOX_ON_TARGET if self.boxes_on_target[i] else self.COLOR_BOX
            pygame.draw.rect(self.screen, color, box)
            pygame.draw.rect(self.screen, self.COLOR_TEXT, box, 2) # Outline

        # Draw ground line
        pygame.draw.line(self.screen, self.COLOR_PLATFORM, (0, self.GROUND_Y), (self.SCREEN_WIDTH, self.GROUND_Y), 3)

        # Draw selector
        if not self.game_over and len(self.boxes) > 0:
            active_box = self.boxes[self.active_box_index]
            p1 = (active_box.centerx, active_box.top - 20)
            p2 = (active_box.centerx - 10, active_box.top - 5)
            p3 = (active_box.centerx + 10, active_box.top - 5)
            pygame.gfxdraw.aapolygon(self.screen, (p1, p2, p3), self.COLOR_SELECTOR)
            pygame.gfxdraw.filled_polygon(self.screen, (p1, p2, p3), self.COLOR_SELECTOR)
    
    def _render_ui(self):
        # Render moves left
        moves_text = self.font_main.render(f"Moves Left: {self.moves_left}", True, self.COLOR_TEXT)
        self.screen.blit(moves_text, (10, 10))
        
        # Render score
        score_text = self.font_main.render(f"Score: {self.score:.1f}", True, self.COLOR_TEXT)
        score_rect = score_text.get_rect(topright=(self.SCREEN_WIDTH - 10, 10))
        self.screen.blit(score_text, score_rect)

        # Render game over message
        if self.game_over:
            if all(self.boxes_on_target):
                msg = "YOU WIN!"
                color = self.COLOR_TEXT_SUCCESS
            else:
                msg = "GAME OVER"
                color = self.COLOR_TEXT_FAIL
            
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((255, 255, 255, 180))
            self.screen.blit(overlay, (0, 0))

            game_over_text = self.font_game_over.render(msg, True, color)
            text_rect = game_over_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(game_over_text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "moves_left": self.moves_left,
            "boxes_on_target": sum(self.boxes_on_target),
        }

    def close(self):
        pygame.font.quit()
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
if __name__ == '__main__':
    import os
    # To run headless, uncomment the next line
    # os.environ["SDL_VIDEODRIVER"] = "dummy"

    env = GameEnv()
    
    # --- Manual Play Loop ---
    obs, info = env.reset()
    done = False
    
    # Create a window to display the game
    pygame.display.set_caption("Box Pusher Puzzle")
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    
    running = True
    clock = pygame.time.Clock()
    
    print(env.user_guide)
    print("Press 'R' to reset the level.")

    while running:
        action_taken = False
        action = [0, 0, 0] # Default to no-op, but we only step on key presses
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    action[0] = 1 # Select next
                    action_taken = True
                elif event.key == pygame.K_DOWN:
                    action[0] = 2 # Select prev
                    action_taken = True
                elif event.key == pygame.K_LEFT:
                    action[0] = 3 # Push left
                    action_taken = True
                elif event.key == pygame.K_RIGHT:
                    action[0] = 4 # Push right
                    action_taken = True
                elif event.key == pygame.K_r: # Reset
                    obs, info = env.reset()
                    done = False
                
                if action_taken and not done:
                    obs, reward, terminated, truncated, info = env.step(action)
                    done = terminated or truncated
                    print(f"Action: {action}, Reward: {reward:.2f}, Info: {info}")
        
        # Render the environment to the screen
        frame = env._get_observation()
        frame_surface = pygame.surfarray.make_surface(np.transpose(frame, (1, 0, 2)))
        screen.blit(frame_surface, (0, 0))
        pygame.display.flip()
        
        clock.tick(30) # Limit FPS

    env.close()