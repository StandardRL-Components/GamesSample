
# Generated: 2025-08-28T05:15:36.307663
# Source Brief: brief_02559.md
# Brief Index: 2559

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
from collections import namedtuple
import os
import pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Data structure for boxes
Box = namedtuple("Box", ["x", "y", "vx", "vy", "id", "in_goal", "color"])

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: ← to push the leftmost available box, → to push the rightmost available box."
    )

    game_description = (
        "Push boxes up the slopes to their designated goal zones. Plan your moves carefully as you have a limited number of pushes."
    )

    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Screen and rendering setup
        self.WIDTH, self.HEIGHT = 640, 400
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.render_mode = render_mode
        
        # Gymnasium spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame initialization
        pygame.init()
        pygame.font.init()
        self.font_ui = pygame.font.Font(None, 28)
        self.font_title = pygame.font.Font(None, 48)

        # Colors
        self.COLOR_BG = (25, 35, 45)
        self.COLOR_SLOPE = (70, 80, 90)
        self.COLOR_SLOPE_SHADOW = (50, 60, 70)
        self.COLOR_GOAL = (100, 220, 120)
        self.COLOR_TEXT = (230, 230, 230)
        self.BOX_COLORS = [
            ((217, 95, 2), (152, 66, 0)),    # Orange
            ((27, 158, 119), (18, 110, 83)), # Teal
            ((231, 41, 138), (161, 28, 96)), # Magenta
            ((117, 112, 179), (82, 78, 125)) # Purple
        ]
        
        # Game constants
        self.MAX_PUSHES = 15
        self.BOX_SIZE = 30
        self.SLOPE_WIDTH = 100
        self.SLOPE_HEIGHT = 200
        self.SLOPE_THICKNESS = 10
        self.START_Y = self.HEIGHT - 50
        
        # Physics constants
        self.PUSH_VELOCITY = 18.0 
        self.GRAVITY_EFFECT = 0.4 # Deceleration along the slope

        # Game state variables
        self.slopes = []
        self.goals = []
        self.boxes = []
        self.pushes_remaining = 0
        self.score = 0
        self.steps = 0
        self.game_over = False
        self.last_push_info = None # For rendering push arrow
        self.last_push_timer = 0
        
        self._setup_level()
        self.reset()
        
        # Final validation check
        self.validate_implementation()

    def _setup_level(self):
        """Initializes the static parts of the level (slopes and goals)."""
        num_slopes = 4
        total_width = num_slopes * self.SLOPE_WIDTH
        spacing = (self.WIDTH - total_width) / (num_slopes + 1)
        
        for i in range(num_slopes):
            start_x = spacing * (i + 1) + self.SLOPE_WIDTH * i
            
            # Slope polygon
            p1 = (start_x, self.START_Y)
            p2 = (start_x + self.SLOPE_WIDTH, self.START_Y - self.SLOPE_HEIGHT)
            p3 = (p2[0] + self.SLOPE_THICKNESS, p2[1])
            p4 = (p1[0] + self.SLOPE_THICKNESS, p1[1])
            self.slopes.append({'poly': (p1, p2, p3, p4), 'angle': math.atan2(self.SLOPE_HEIGHT, self.SLOPE_WIDTH)})
            
            # Goal rectangle
            goal_height = self.BOX_SIZE * 1.5
            goal_rect = pygame.Rect(
                p2[0] - self.BOX_SIZE, 
                p2[1], 
                self.BOX_SIZE * 1.2, 
                goal_height
            )
            self.goals.append(goal_rect)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.pushes_remaining = self.MAX_PUSHES
        self.score = 0
        self.steps = 0
        self.game_over = False
        self.last_push_info = None
        self.last_push_timer = 0
        
        self.boxes = []
        box_ids = list(range(len(self.slopes)))
        self.np_random.shuffle(box_ids)

        for i, slope in enumerate(self.slopes):
            start_x = slope['poly'][0][0] + self.SLOPE_THICKNESS / 2
            box_id = box_ids[i]
            self.boxes.append(
                {
                    "id": box_id,
                    "x": start_x + self.SLOPE_WIDTH / 2 - self.BOX_SIZE / 2,
                    "y": self.START_Y - self.BOX_SIZE,
                    "in_goal": False,
                    "color": self.BOX_COLORS[box_id][0],
                    "shadow_color": self.BOX_COLORS[box_id][1],
                    "slope_idx": i
                }
            )
            
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        movement = action[0]
        reward = 0
        terminated = False
        
        # Decay the push arrow animation timer
        if self.last_push_timer > 0:
            self.last_push_timer -= 1
            if self.last_push_timer == 0:
                self.last_push_info = None

        # A push action is taken (left or right)
        if movement in [3, 4]:
            target_box = self._find_pushable_box(movement)
            
            if target_box:
                # --- State before push ---
                old_distances = {b['id']: self._dist_to_goal(b) for b in self.boxes}
                old_in_goal_state = {b['id']: b['in_goal'] for b in self.boxes}
                
                self.pushes_remaining -= 1
                # SFX: Push sound
                
                # --- Physics Simulation ---
                self._simulate_push(target_box)
                
                # --- State after push ---
                new_distances = {b['id']: self._dist_to_goal(b) for b in self.boxes}
                
                # --- Reward Calculation ---
                # Distance-based reward for the pushed box
                dist_change = old_distances[target_box['id']] - new_distances[target_box['id']]
                if dist_change > 1: # Epsilon to avoid float noise
                    reward += 1.0  # Moved closer
                elif dist_change < -1:
                    reward -= 0.1 # Moved further (should not happen)
                    
                # Event-based reward for entering a goal
                if not old_in_goal_state[target_box['id']] and target_box['in_goal']:
                    reward += 5.0
                    # SFX: Goal success chime
                
                # Update score
                self.score += reward
                
                # Set up push animation
                self.last_push_info = {'x': target_box['x'], 'y': target_box['y']}
                self.last_push_timer = 5 # frames
            
        # --- Termination Checks ---
        all_in_goal = all(b['in_goal'] for b in self.boxes)
        
        if all_in_goal:
            terminated = True
            self.game_over = True
            reward += 100.0
            self.score += 100.0
            # SFX: Level complete fanfare
        elif self.pushes_remaining <= 0:
            terminated = True
            self.game_over = True
            reward -= 100.0
            self.score -= 100.0
            # SFX: Failure sound
            
        self.steps += 1
        if self.steps >= 1000: # Max episode length fallback
            terminated = True
            self.game_over = True

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _find_pushable_box(self, direction):
        """Finds the correct box to push based on action."""
        pushable_boxes = [b for b in self.boxes if int(b['y']) == int(self.START_Y - self.BOX_SIZE)]
        
        if not pushable_boxes:
            return None
            
        if direction == 3: # Left
            return min(pushable_boxes, key=lambda b: b['x'])
        elif direction == 4: # Right
            return max(pushable_boxes, key=lambda b: b['x'])
        return None

    def _simulate_push(self, box):
        """Calculates the final position of a box after a push in a single step."""
        slope = self.slopes[box['slope_idx']]
        goal = self.goals[box['slope_idx']]
        
        # 1D physics along the slope
        velocity = self.PUSH_VELOCITY
        dist_traveled = 0
        slope_len = np.linalg.norm(np.array(slope['poly'][1][:2]) - np.array(slope['poly'][0][:2]))

        while velocity > 0:
            velocity -= self.GRAVITY_EFFECT
            if velocity < 0: velocity = 0
            dist_traveled += velocity

            # Check for collision with top of slope
            if dist_traveled >= slope_len:
                dist_traveled = slope_len
                break
        
        # Convert 1D distance back to 2D coordinates
        start_pos = np.array((slope['poly'][0][0] + self.SLOPE_THICKNESS / 2, slope['poly'][0][1]))
        slope_vec = np.array(slope['poly'][1][:2]) - np.array(slope['poly'][0][:2])
        slope_unit_vec = slope_vec / np.linalg.norm(slope_vec)
        
        final_pos_on_slope = start_pos + slope_unit_vec * dist_traveled
        
        box['x'] = final_pos_on_slope[0] - self.BOX_SIZE / 2
        box['y'] = final_pos_on_slope[1] - self.BOX_SIZE
        
        # Check if in goal
        box_rect = pygame.Rect(box['x'], box['y'], self.BOX_SIZE, self.BOX_SIZE)
        if box_rect.colliderect(goal):
            box['in_goal'] = True
        else:
            box['in_goal'] = False

    def _dist_to_goal(self, box):
        goal_center = self.goals[box['slope_idx']].center
        box_center = (box['x'] + self.BOX_SIZE/2, box['y'] + self.BOX_SIZE/2)
        return math.hypot(goal_center[0] - box_center[0], goal_center[1] - box_center[1])

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
            "pushes_remaining": self.pushes_remaining,
            "boxes_in_goal": sum(1 for b in self.boxes if b['in_goal'])
        }

    def _render_game(self):
        # Draw slopes and goals (background elements)
        for i, slope in enumerate(self.slopes):
            # Goal Zone
            goal_rect = self.goals[i]
            # Use a transparent surface for the goal
            goal_surface = pygame.Surface(goal_rect.size, pygame.SRCALPHA)
            goal_color = self.COLOR_GOAL if not all(b['in_goal'] for b in self.boxes) else (255, 215, 0)
            goal_surface.fill(goal_color + (60,))
            self.screen.blit(goal_surface, goal_rect.topleft)
            pygame.draw.rect(self.screen, goal_color, goal_rect, 1)

            # Slope
            pygame.draw.polygon(self.screen, self.COLOR_SLOPE, slope['poly'])
            # Shadow for 3D effect
            p1, p2, p3, p4 = slope['poly']
            pygame.draw.polygon(self.screen, self.COLOR_SLOPE_SHADOW, (p1, p4, (p4[0], p4[1]+5), (p1[0], p1[1]+5)))


        # Draw boxes (interactive elements)
        for box in sorted(self.boxes, key=lambda b: b['y']): # Draw from back to front
            rect = pygame.Rect(int(box['x']), int(box['y']), self.BOX_SIZE, self.BOX_SIZE)
            # 3D effect
            pygame.draw.rect(self.screen, box['shadow_color'], (rect.left + 4, rect.top + 4, rect.width, rect.height))
            pygame.draw.rect(self.screen, box['color'], rect)
            pygame.draw.rect(self.screen, (255, 255, 255), rect, 1) # Outline
            
        # Draw push animation
        if self.last_push_info:
            alpha = int(255 * (self.last_push_timer / 5.0))
            x, y = self.last_push_info['x'], self.last_push_info['y']
            p1 = (x + self.BOX_SIZE / 2, y + self.BOX_SIZE + 5)
            p2 = (p1[0] - 8, p1[1] + 12)
            p3 = (p1[0] + 8, p1[1] + 12)
            pygame.gfxdraw.aatrigon(self.screen, int(p1[0]), int(p1[1]), int(p2[0]), int(p2[1]), int(p3[0]), int(p3[1]), (255, 255, 255, alpha))
            pygame.gfxdraw.filled_trigon(self.screen, int(p1[0]), int(p1[1]), int(p2[0]), int(p2[1]), int(p3[0]), int(p3[1]), (255, 255, 255, alpha))

    def _render_ui(self):
        # Pushes Remaining
        pushes_text = self.font_ui.render(f"Pushes: {self.pushes_remaining}", True, self.COLOR_TEXT)
        self.screen.blit(pushes_text, (15, 15))
        
        # Score
        score_text = self.font_ui.render(f"Score: {int(self.score)}", True, self.COLOR_TEXT)
        score_rect = score_text.get_rect(topright=(self.WIDTH - 15, 15))
        self.screen.blit(score_text, score_rect)

        # Game Over Text
        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            self.screen.blit(overlay, (0, 0))
            
            won = all(b['in_goal'] for b in self.boxes)
            end_text_str = "PUZZLE SOLVED!" if won else "OUT OF PUSHES"
            end_color = self.COLOR_GOAL if won else (220, 50, 50)
            
            end_text = self.font_title.render(end_text_str, True, end_color)
            end_rect = end_text.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2))
            self.screen.blit(end_text, end_rect)

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        """
        Call this at the end of __init__ to verify implementation.
        """
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
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Slope Pusher")
    
    done = False
    total_reward = 0
    
    print("\n" + "="*30)
    print("Slope Pusher - Manual Test")
    print("="*30)
    print(env.user_guide)
    print("Press Q to quit.")
    
    while not done:
        action = [0, 0, 0] # Default action: no-op
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    done = True
                if event.key == pygame.K_LEFT:
                    action[0] = 3
                if event.key == pygame.K_RIGHT:
                    action[0] = 4
                if event.key == pygame.K_r: # Reset game
                    obs, info = env.reset()
                    total_reward = 0
                    print("--- Game Reset ---")

        # Only step if a key was pressed (turn-based)
        if action[0] != 0:
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            print(f"Action: {action}, Reward: {reward:.2f}, Total Reward: {total_reward:.2f}, Pushes: {info['pushes_remaining']}")
            
            if terminated or truncated:
                print("--- Episode Finished ---")
                print(f"Final Score: {info['score']}, Total Steps: {info['steps']}")
                done = True

        # Render the observation to the screen
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        env.clock.tick(30) # Limit FPS for human play
        
    env.close()