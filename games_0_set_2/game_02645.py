
# Generated: 2025-08-28T05:31:41.171683
# Source Brief: brief_02645.md
# Brief Index: 2645

        
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
        "Controls: Press space to flip gravity. Your goal is to get all boxes onto the targets."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A physics-based puzzle game. Manipulate gravity to guide colored boxes to their matching targets. "
        "Solve the puzzle in as few moves as possible."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Screen and world dimensions
        self.WIDTH, self.HEIGHT = 640, 400
        self.NUM_BOXES = 7
        self.MOVE_LIMIT = 25
        self.BOX_SIZE = 30
        self.GRAVITY_STRENGTH = 0.98

        # EXACT spaces:
        self.observation_space = Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_main = pygame.font.Font(None, 36)
        self.font_small = pygame.font.Font(None, 24)

        # Colors
        self.COLOR_BG = (20, 25, 40)
        self.COLOR_GRID = (35, 40, 60)
        self.COLOR_TARGET = (70, 80, 100)
        self.COLOR_TEXT = (220, 230, 255)
        self.COLOR_ARROW = (255, 255, 255)
        
        self.BOX_COLORS = [
            (255, 80, 80),   # Red
            (80, 255, 80),   # Green
            (80, 120, 255),  # Blue
            (255, 255, 80),  # Yellow
            (200, 80, 255),  # Purple
            (255, 150, 80),  # Orange
            (80, 255, 255),  # Cyan
        ]
        self.BOX_COLORS_LIT = [tuple(min(255, int(c * 1.5)) for c in color) for color in self.BOX_COLORS]
        self.BOX_COLORS_DARK = [tuple(int(c * 0.5) for c in color) for color in self.BOX_COLORS]

        # State variables (initialized in reset)
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.moves = 0
        self.gravity_direction = 0
        self.boxes = []
        self.targets = []
        self.last_space_held = False
        self.previous_box_distances = []
        self.last_on_target_count = 0

        self.reset()
        self.validate_implementation()

    def _initialize_positions(self):
        """Sets up the initial state of boxes and targets for a new game."""
        self.targets = []
        self.boxes = []

        # Define fixed target locations
        target_positions = [
            (50, self.HEIGHT - 40), (150, self.HEIGHT - 40), (250, self.HEIGHT - 40),
            (350, 40), (450, 40), (550, 40),
            (self.WIDTH / 2 - self.BOX_SIZE / 2, self.HEIGHT / 2 - self.BOX_SIZE / 2)
        ]
        
        for i in range(self.NUM_BOXES):
            pos = target_positions[i]
            self.targets.append(pygame.Rect(pos[0], pos[1] - self.BOX_SIZE, self.BOX_SIZE, self.BOX_SIZE))
            
        # Place boxes in starting positions, avoiding overlap
        placed_boxes = []
        attempts = 0
        while len(self.boxes) < self.NUM_BOXES and attempts < 1000:
            attempts += 1
            start_x = self.np_random.integers(self.BOX_SIZE, self.WIDTH - self.BOX_SIZE)
            start_y = self.np_random.choice([self.BOX_SIZE, self.HEIGHT - self.BOX_SIZE * 2])
            new_box_rect = pygame.Rect(start_x, start_y, self.BOX_SIZE, self.BOX_SIZE)
            
            is_overlapping = False
            for box in self.boxes:
                if new_box_rect.colliderect(box['rect']):
                    is_overlapping = True
                    break
            
            if not is_overlapping:
                self.boxes.append({
                    'rect': new_box_rect,
                    'vel': pygame.Vector2(0, 0),
                    'color_idx': len(self.boxes),
                    'on_target': False,
                    'settled': True
                })

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.moves = 0
        self.gravity_direction = 1  # 1 for down, -1 for up
        self.last_space_held = False
        self.last_on_target_count = 0

        self._initialize_positions()
        
        self.previous_box_distances = [self._get_dist_to_closest_target(box) for box in self.boxes]

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        action_taken = False
        reward = 0

        # Detect a "press" of the space button to flip gravity
        if space_held and not self.last_space_held:
            # Player used a move
            action_taken = True
            self.moves += 1
            self.gravity_direction *= -1
            
            # Unsettle all boxes so they can move
            for box in self.boxes:
                box['settled'] = False
            
            # Sound effect placeholder
            # play_sound('gravity_shift')

        self.last_space_held = space_held
        
        # If an action was taken, simulate physics until all boxes settle
        if action_taken:
            self._update_physics()
            reward = self._calculate_reward()
            self._check_termination()

        self.steps += 1
        self.score += reward

        return self._get_observation(), reward, self.game_over, False, self._get_info()

    def _update_physics(self):
        """Simulates box movement until all boxes have stopped."""
        sim_steps = 0
        max_sim_steps = 300  # Safety break for the simulation loop
        
        while sim_steps < max_sim_steps:
            boxes_moving = 0
            
            # Sort boxes by their vertical position relative to gravity direction
            # This ensures correct stacking behavior in one pass
            sorted_boxes = sorted(self.boxes, key=lambda b: b['rect'].y, reverse=(self.gravity_direction == 1))

            for box in sorted_boxes:
                if box['settled']:
                    continue

                # Store old position to check for movement
                old_y = box['rect'].y

                # Apply gravity
                box['vel'].y += self.gravity_direction * self.GRAVITY_STRENGTH
                box['rect'].y += box['vel'].y

                # Boundary checks (floor and ceiling)
                if box['rect'].bottom > self.HEIGHT:
                    box['rect'].bottom = self.HEIGHT
                    box['vel'].y = 0
                elif box['rect'].top < 0:
                    box['rect'].top = 0
                    box['vel'].y = 0

                # Collision with other boxes
                for other_box in self.boxes:
                    if box is other_box:
                        continue
                    if box['rect'].colliderect(other_box['rect']):
                        if self.gravity_direction == 1: # Falling down
                            box['rect'].bottom = other_box['rect'].top
                        else: # Falling up
                            box['rect'].top = other_box['rect'].bottom
                        box['vel'].y = 0

                # Check if the box has stopped moving
                if abs(old_y - box['rect'].y) < 0.1:
                    box['settled'] = True
                else:
                    boxes_moving += 1
            
            if boxes_moving == 0:
                break # All boxes have settled
            
            sim_steps += 1

    def _get_dist_to_closest_target(self, box):
        """Calculates the distance from a box to its closest valid target."""
        box_center = pygame.Vector2(box['rect'].center)
        min_dist = float('inf')
        for target in self.targets:
            target_center = pygame.Vector2(target.center)
            min_dist = min(min_dist, box_center.distance_to(target_center))
        return min_dist

    def _calculate_reward(self):
        """Calculates the reward for the current state."""
        if self.game_over:
            if all(b['on_target'] for b in self.boxes):
                return 100.0  # Win
            else:
                return -100.0 # Loss
        
        reward = 0.0
        current_on_target_count = 0
        
        for i, box in enumerate(self.boxes):
            dist = self._get_dist_to_closest_target(box)
            
            # Reward for moving closer to a target
            if dist < self.previous_box_distances[i]:
                reward += 1.0
            elif dist > self.previous_box_distances[i]:
                reward -= 0.1
            
            self.previous_box_distances[i] = dist
            
            # Check if box is on any target
            is_on_target = False
            for target in self.targets:
                if target.contains(box['rect']):
                    is_on_target = True
                    break
            box['on_target'] = is_on_target
            if is_on_target:
                current_on_target_count += 1
        
        # Reward for placing a new box on a target
        if current_on_target_count > self.last_on_target_count:
            reward += 5.0 * (current_on_target_count - self.last_on_target_count)
            # Sound effect placeholder
            # play_sound('target_achieved')
            
        self.last_on_target_count = current_on_target_count
        
        return reward

    def _check_termination(self):
        """Checks for win/loss conditions and sets self.game_over."""
        # Win condition
        if all(b['on_target'] for b in self.boxes):
            self.game_over = True
            self.score += 100 # Add final win bonus to score
            return

        # Loss condition
        if self.moves >= self.MOVE_LIMIT:
            self.game_over = True
            self.score -= 100 # Add final loss penalty to score
            return
            
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "moves": self.moves,
            "boxes_on_target": self.last_on_target_count,
        }

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw grid
        for x in range(0, self.WIDTH, 40):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.HEIGHT), 1)
        for y in range(0, self.HEIGHT, 40):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.WIDTH, y), 1)

        # Draw targets
        for target in self.targets:
            pygame.draw.rect(self.screen, self.COLOR_TARGET, target)

        # Draw boxes
        for box in self.boxes:
            color_idx = box['color_idx']
            main_color = self.BOX_COLORS_LIT[color_idx] if box['on_target'] else self.BOX_COLORS[color_idx]
            dark_color = self.BOX_COLORS_DARK[color_idx]
            
            # Draw outline
            outline_rect = box['rect'].inflate(4, 4)
            pygame.draw.rect(self.screen, dark_color, outline_rect, border_radius=4)
            # Draw main box
            pygame.draw.rect(self.screen, main_color, box['rect'], border_radius=3)

    def _render_ui(self):
        # Render move counter
        move_text = f"Moves: {self.moves}/{self.MOVE_LIMIT}"
        text_surface = self.font_main.render(move_text, True, self.COLOR_TEXT)
        self.screen.blit(text_surface, (15, 10))
        
        # Render boxes on target counter
        target_text = f"On Target: {self.last_on_target_count}/{self.NUM_BOXES}"
        target_surface = self.font_small.render(target_text, True, self.COLOR_TEXT)
        self.screen.blit(target_surface, (15, 45))

        # Render gravity indicator arrow
        arrow_center_x = self.WIDTH - 30
        arrow_center_y = self.HEIGHT / 2
        if self.gravity_direction == 1: # Down
            points = [(arrow_center_x, arrow_center_y + 15), 
                      (arrow_center_x - 10, arrow_center_y - 5), 
                      (arrow_center_x + 10, arrow_center_y - 5)]
        else: # Up
            points = [(arrow_center_x, arrow_center_y - 15), 
                      (arrow_center_x - 10, arrow_center_y + 5), 
                      (arrow_center_x + 10, arrow_center_y + 5)]
        pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_ARROW)
        pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_ARROW)

        # Render game over message
        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            win_condition = all(b['on_target'] for b in self.boxes)
            msg = "PUZZLE SOLVED!" if win_condition else "OUT OF MOVES"
            color = (100, 255, 100) if win_condition else (255, 100, 100)
            
            msg_surface = self.font_main.render(msg, True, color)
            msg_rect = msg_surface.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
            self.screen.blit(msg_surface, msg_rect)

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

# Example of how to run the environment
if __name__ == '__main__':
    env = GameEnv()
    
    # --- Manual Play Example ---
    # This part requires a display. If running headlessly, comment this out.
    try:
        import os
        os.environ["SDL_VIDEODRIVER"] = "dummy" # Force headless for servers
        pygame.display.init()
        screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
        pygame.display.set_caption(env.game_description)
        os.environ["SDL_VIDEODRIVER"] = "" # Allow display for local testing
    except pygame.error:
        print("No display available. Running in headless mode.")
        screen = None

    if screen:
        obs, info = env.reset()
        done = False
        clock = pygame.time.Clock()
        
        print(env.user_guide)

        while not done:
            action = [0, 0, 0] # Default no-op
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    done = True
                if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                    obs, info = env.reset() # Reset on 'r' key

            keys = pygame.key.get_pressed()
            if keys[pygame.K_SPACE]:
                action[1] = 1 # Space pressed

            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # Render to the screen
            surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
            screen.blit(surf, (0, 0))
            pygame.display.flip()

            clock.tick(30) # Limit frame rate for playability

        print(f"Game Over. Final Score: {info['score']}")
        pygame.quit()