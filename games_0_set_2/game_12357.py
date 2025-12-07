import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T14:05:06.944187
# Source Brief: brief_02357.md
# Brief Index: 2357
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class Branch:
    """A helper class to manage a single growing tip of the fractal."""
    def __init__(self, start_pos, angle, level):
        self.start_pos = np.array(start_pos, dtype=float)
        self.end_pos = np.array(start_pos, dtype=float)
        self.angle = angle  # in degrees
        self.level = level
        self.growth_progress = 0.0

class GameEnv(gym.Env):
    """
    A Gymnasium environment where the agent guides a recursively branching
    fractal line through a field of obstacles. The goal is to achieve
    maximum growth without collision.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Guide the growth of a fractal tree, navigating through a field of obstacles "
        "to reach its full potential without collision."
    )
    user_guide = (
        "Controls: Use ← and → arrow keys to change the branching angle of the fractal."
    )
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Constants ---
        self.SCREEN_WIDTH = 640
        self.SCREEN_HEIGHT = 400
        self.FPS = 30
        self.MAX_SEGMENTS_WIN = 50
        self.MAX_STEPS = 5000
        self.SEGMENT_LENGTH = 10.0
        self.OBSTACLE_RADIUS = 5
        self.OBSTACLE_FADE_IN_STEPS = 15 # 0.5 seconds at 30 FPS

        # --- Colors ---
        self.COLOR_BG = (10, 10, 20)
        self.COLOR_FRACTAL = (255, 255, 255)
        self.COLOR_OBSTACLE = (255, 50, 50)
        self.COLOR_UI = (220, 220, 220)

        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("monospace", 18, bold=True)
        self.font_game_over = pygame.font.SysFont("monospace", 48, bold=True)

        # --- Game State (initialized in reset) ---
        self.steps = 0
        self.score = 0.0
        self.game_over = False
        self.branch_angle_deg = 0.0
        self.segments = []
        self.active_branches = []
        self.obstacles = []
        # self.reset() # reset is called by the environment wrapper
        # self.validate_implementation() # this is for debugging and should not be in the final class

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Initialize all game state
        self.steps = 0
        self.score = 0.0
        self.game_over = False
        self.branch_angle_deg = 0.0

        # Reset fractal
        self.segments = []
        self.active_branches = []
        initial_branch = Branch(
            start_pos=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT),
            angle=-90,  # Pointing straight up
            level=0
        )
        self.active_branches.append(initial_branch)

        # Reset obstacles
        self._generate_obstacles()

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0.0, True, False, self._get_info()

        self.steps += 1
        reward = 0.0

        # 1. Handle player action
        self._handle_action(action)

        # 2. Update game state (fractal growth)
        new_segments_count, collision_detected = self._update_fractal()
        reward += new_segments_count * 1.0  # +1 for each new segment

        # 3. Check for termination and assign rewards
        terminated = False
        truncated = False
        if collision_detected:
            reward = -100.0
            terminated = True
            # Sound: Collision/Failure
        elif len(self.segments) >= self.MAX_SEGMENTS_WIN:
            reward = 100.0
            terminated = True
            # Sound: Victory
        elif self.steps >= self.MAX_STEPS:
            truncated = True

        if not terminated and not truncated:
            reward += 0.1  # +0.1 survival reward per step

        self.score += reward
        self.game_over = terminated or truncated

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _handle_action(self, action):
        movement = action[0]
        # Action 3 (left): Decrease branching angle
        if movement == 3:
            self.branch_angle_deg -= 1.0
            # Sound: UI tick down
        # Action 4 (right): Increase branching angle
        elif movement == 4:
            self.branch_angle_deg += 1.0
            # Sound: UI tick up

        self.branch_angle_deg = np.clip(self.branch_angle_deg, -45, 45)

    def _update_fractal(self):
        newly_created_branches = []
        collision_found = False
        new_segments_this_step = 0

        for branch in self.active_branches:
            growth_speed = (self.SEGMENT_LENGTH / self.FPS) * (0.95 ** branch.level)
            
            old_end_pos = branch.end_pos.copy()
            
            rad = math.radians(branch.angle)
            direction_vec = np.array([math.cos(rad), math.sin(rad)])
            
            new_growth = direction_vec * growth_speed
            branch.end_pos += new_growth
            branch.growth_progress += growth_speed
            
            if self._check_line_obstacle_collision(old_end_pos, branch.end_pos):
                collision_found = True
                self.segments.append(branch)
                break 

            if branch.growth_progress >= self.SEGMENT_LENGTH:
                # Finalize segment position to exact length
                branch.end_pos = branch.start_pos + direction_vec * self.SEGMENT_LENGTH
                self.segments.append(branch)
                new_segments_this_step += 1
                # Sound: Branch created
                
                if len(self.segments) < self.MAX_SEGMENTS_WIN:
                    # Spawn two new branches
                    new_level = branch.level + 1
                    branch1 = Branch(branch.end_pos, branch.angle + self.branch_angle_deg, new_level)
                    branch2 = Branch(branch.end_pos, branch.angle - self.branch_angle_deg, new_level)
                    newly_created_branches.extend([branch1, branch2])
            else:
                # Branch is still growing, keep it for the next frame
                newly_created_branches.append(branch)
        
        if not collision_found:
            self.active_branches = newly_created_branches
        else:
            self.active_branches = [] # Stop all growth on collision

        return new_segments_this_step, collision_found

    def _generate_obstacles(self):
        self.obstacles = []
        obstacle_area = math.pi * (self.OBSTACLE_RADIUS ** 2)
        target_coverage = self.SCREEN_WIDTH * self.SCREEN_HEIGHT * 0.10
        num_obstacles = int(target_coverage / obstacle_area)
        
        start_safe_zone_radius_sq = 100**2

        for _ in range(num_obstacles):
            while True:
                pos = np.array([
                    self.np_random.uniform(self.OBSTACLE_RADIUS, self.SCREEN_WIDTH - self.OBSTACLE_RADIUS),
                    self.np_random.uniform(self.OBSTACLE_RADIUS, self.SCREEN_HEIGHT - self.OBSTACLE_RADIUS - 50) # Keep top clear
                ])
                start_pos = np.array([self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT])
                dist_sq = np.sum((pos - start_pos)**2)
                if dist_sq > start_safe_zone_radius_sq:
                    break
            
            self.obstacles.append({
                'pos': pos,
                'radius': self.OBSTACLE_RADIUS,
                'creation_step': self.steps
            })

    def _check_line_obstacle_collision(self, p1, p2):
        for obs in self.obstacles:
            # Only check collision with fully faded-in (solid) obstacles
            fade_progress = (self.steps - obs['creation_step']) / self.OBSTACLE_FADE_IN_STEPS
            if fade_progress < 1.0:
                continue

            c = obs['pos']
            r = obs['radius']
            
            line_vec = p2 - p1
            p1_to_c = c - p1
            
            line_len_sq = np.dot(line_vec, line_vec)
            if line_len_sq == 0.0: # p1 and p2 are the same point
                if np.dot(p1_to_c, p1_to_c) < r**2:
                    return True
                continue

            t = np.dot(p1_to_c, line_vec) / line_len_sq
            t = np.clip(t, 0, 1)
            
            closest_point = p1 + t * line_vec
            dist_sq = np.dot(closest_point - c, closest_point - c)

            if dist_sq < r**2:
                return True
        return False

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Render obstacles
        for obs in self.obstacles:
            fade_progress = min(1.0, (self.steps - obs['creation_step']) / self.OBSTACLE_FADE_IN_STEPS)
            color = (*self.COLOR_OBSTACLE, int(255 * fade_progress))
            pygame.gfxdraw.filled_circle(
                self.screen, int(obs['pos'][0]), int(obs['pos'][1]), int(obs['radius']), color
            )
            pygame.gfxdraw.aacircle(
                self.screen, int(obs['pos'][0]), int(obs['pos'][1]), int(obs['radius']), color
            )

        # Render completed fractal segments
        for seg in self.segments:
            pygame.draw.aaline(
                self.screen, self.COLOR_FRACTAL, seg.start_pos, seg.end_pos, 1
            )
            
        # Render active, growing branches
        for branch in self.active_branches:
            pygame.draw.aaline(
                self.screen, self.COLOR_FRACTAL, branch.start_pos, branch.end_pos, 1
            )

    def _render_ui(self):
        # Segments count
        segment_text = f"Segments: {len(self.segments)} / {self.MAX_SEGMENTS_WIN}"
        text_surface = self.font_ui.render(segment_text, True, self.COLOR_UI)
        self.screen.blit(text_surface, (10, 10))

        # Branching angle
        angle_text = f"Angle: {self.branch_angle_deg:.1f}°"
        text_surface = self.font_ui.render(angle_text, True, self.COLOR_UI)
        text_rect = text_surface.get_rect(topright=(self.SCREEN_WIDTH - 10, 10))
        self.screen.blit(text_surface, text_rect)

        # Game Over message
        if self.game_over:
            if len(self.segments) >= self.MAX_SEGMENTS_WIN:
                msg = "GROWTH COMPLETE"
                color = (100, 255, 100)
            else:
                msg = "COLLISION"
                color = (255, 100, 100)
            
            go_surface = self.font_game_over.render(msg, True, color)
            go_rect = go_surface.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(go_surface, go_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "segments": len(self.segments),
            "branch_angle": self.branch_angle_deg,
        }

    def close(self):
        pygame.quit()


if __name__ == '__main__':
    # This block allows you to play the game manually for testing
    # It will not run in the evaluation environment
    os.environ["SDL_VIDEODRIVER"] = "x11" # Use a display driver
    env = GameEnv()
    obs, info = env.reset()
    
    running = True
    terminated = False
    truncated = False
    
    # Use a window to display the game
    render_screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    pygame.display.set_caption("Fractal Growth Environment")
    
    action = np.array([0, 0, 0]) # [movement, space, shift]
    
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        # Manual controls
        keys = pygame.key.get_pressed()
        action.fill(0) # Reset actions
        if keys[pygame.K_LEFT]:
            action[0] = 3
        elif keys[pygame.K_RIGHT]:
            action[0] = 4
        else:
            action[0] = 0

        if not (terminated or truncated):
            obs, reward, terminated, truncated, info = env.step(action)
            print(f"Step: {info['steps']}, Segments: {info['segments']}, Reward: {reward:.2f}, Score: {info['score']:.2f}")

        # Render the observation to the display window
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        render_screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        env.clock.tick(env.FPS)
        
        if terminated or truncated:
            pygame.time.wait(2000) # Pause for 2 seconds on game over
            obs, info = env.reset()
            terminated = False
            truncated = False

    env.close()