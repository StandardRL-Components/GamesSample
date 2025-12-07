import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T11:56:43.444830
# Source Brief: brief_00780.md
# Brief Index: 780
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

# Helper class for particles for visual effects
class Particle:
    """A simple particle for visual effects like explosions."""
    def __init__(self, pos, color, lifetime, radius, velocity):
        self.pos = pygame.Vector2(pos)
        self.color = color
        self.lifetime = lifetime
        self.radius = radius
        self.velocity = pygame.Vector2(velocity)
        self.gravity = pygame.Vector2(0, 0.08) # Gentle gravity

    def update(self):
        """Updates the particle's position, velocity, and lifetime."""
        self.pos += self.velocity
        self.velocity += self.gravity
        self.lifetime -= 1
        self.radius = max(0, self.radius - 0.05)

    def draw(self, surface):
        """Draws the particle as a circle."""
        if self.lifetime > 0:
            pygame.draw.circle(surface, self.color, (int(self.pos.x), int(self.pos.y)), int(self.radius))

# Helper class for tree branches
class Branch:
    """Represents a single branch of the fractal tree."""
    def __init__(self, start_pos, angle, length, width, generation, grow_speed, target_pos):
        self.start_pos = pygame.Vector2(start_pos)
        self.angle = angle
        self.max_length = length
        self.width = max(1, width)
        self.generation = generation
        self.grow_speed = grow_speed
        self.target_pos = pygame.Vector2(target_pos) if target_pos else None

        self.current_length = 0.0
        self.end_pos = self.start_pos + pygame.Vector2(0, -self.current_length).rotate(self.angle)
        
        self.is_growing = True
        self.is_tip = True
        self.color = (100, 255, 150)

    def update(self):
        """Grows the branch and returns the change in distance to its target."""
        if not self.is_growing:
            return 0.0

        dist_before = float('inf')
        if self.target_pos:
            dist_before = self.end_pos.distance_to(self.target_pos)

        self.current_length += self.grow_speed
        if self.current_length >= self.max_length:
            self.current_length = self.max_length
            self.is_growing = False

        self.end_pos = self.start_pos + pygame.Vector2(0, -self.current_length).rotate(self.angle)

        dist_after = float('inf')
        if self.target_pos:
            dist_after = self.end_pos.distance_to(self.target_pos)
        
        return dist_before - dist_after

    def draw(self, surface):
        """Draws the branch as a thick line with a rounded cap."""
        if self.current_length > 1:
            start = self.start_pos
            end = self.end_pos
            width = int(self.width)
            
            # Draw main line
            pygame.draw.line(surface, self.color, (int(start.x), int(start.y)), (int(end.x), int(end.y)), width)
            
            # Draw rounded caps for a smoother look
            pygame.draw.circle(surface, self.color, (int(start.x), int(start.y)), width // 2)
            pygame.draw.circle(surface, self.color, (int(end.x), int(end.y)), width // 2)


class GameEnv(gym.Env):
    """
    A Gymnasium environment where the agent grows a fractal tree to collect resources.
    The goal is to maximize the score by collecting resources within a time limit.
    Visuals and game feel are prioritized.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = "Grow a fractal tree by splitting branches to reach and collect floating resources before time runs out."
    user_guide = "Use the arrow keys (↑↓←→) to select the highest, lowest, leftmost, or rightmost branch tip to split towards nearby resources."
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    GAME_DURATION_SECONDS = 60
    FPS = 30
    MAX_STEPS = GAME_DURATION_SECONDS * FPS
    ACTION_COOLDOWN_PERIOD = 1 * FPS # 1 action per second
    
    NUM_RESOURCES = 15
    RESOURCE_RADIUS = 10
    RESOURCE_COLLECT_RADIUS = 15

    # Colors
    COLOR_BG_START = (10, 0, 20)
    COLOR_BG_END = (40, 20, 80)
    COLOR_RESOURCE = (50, 150, 255)
    COLOR_RESOURCE_OUTLINE = (150, 220, 255)
    COLOR_UI_TEXT = (240, 240, 240)
    COLOR_TIMER_WARN = (255, 100, 100)
    COLOR_SELECTOR = (255, 255, 0)

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Gymnasium spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("monospace", 20, bold=True)
        self.font_timer = pygame.font.SysFont("monospace", 24, bold=True)
        
        # Pre-render background for performance
        self.background_surface = self._create_gradient_background()
        
        # Initialize state variables
        self.branches = []
        self.resources = []
        self.particles = []
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.action_cooldown = 0
        self.last_action_tip_idx = -1
        self.last_action_feedback_timer = 0
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.action_cooldown = 0
        self.last_action_tip_idx = -1
        self.last_action_feedback_timer = 0
        
        self.branches.clear()
        self.resources.clear()
        self.particles.clear()
        
        # Initialize tree trunk
        trunk_start = (self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT)
        initial_length = 80
        initial_angle = 0 # Straight up
        initial_width = 12
        trunk_target = (trunk_start[0], trunk_start[1] - initial_length)
        self.branches.append(Branch(trunk_start, initial_angle, initial_length, initial_width, 0, 2.0, trunk_target))

        # Initialize resources
        for _ in range(self.NUM_RESOURCES):
            x = self.np_random.uniform(self.RESOURCE_RADIUS * 2, self.SCREEN_WIDTH - self.RESOURCE_RADIUS * 2)
            y = self.np_random.uniform(self.RESOURCE_RADIUS * 2, self.SCREEN_HEIGHT - 100)
            self.resources.append(pygame.Vector2(x, y))
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        reward = 0.0
        movement = action[0]
        
        self.action_cooldown = max(0, self.action_cooldown - 1)
        self.last_action_feedback_timer = max(0, self.last_action_feedback_timer - 1)

        if movement != 0 and self.action_cooldown == 0:
            action_taken = self._handle_split_action(movement)
            if action_taken:
                self.action_cooldown = self.ACTION_COOLDOWN_PERIOD
                # sfx: branch_split.play()
        
        shaping_reward, collection_reward = self._update_world_state()
        reward += shaping_reward + collection_reward
        
        self.steps += 1
        terminated = self.steps >= self.MAX_STEPS or (len(self.resources) == 0 and not any(b.is_growing for b in self.branches))
        if terminated and not self.game_over:
             reward += self.score * 10 # Final bonus for each resource collected
             self.game_over = True
        
        truncated = False
        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _handle_split_action(self, movement):
        """Finds a branch tip based on player input and splits it."""
        tips = [(i, b) for i, b in enumerate(self.branches) if b.is_tip and not b.is_growing]
        if not tips:
            return False

        selected_tip_idx = -1
        try:
            if movement == 1: # Up
                selected_tip_idx, _ = min(tips, key=lambda item: item[1].end_pos.y)
            elif movement == 2: # Down
                selected_tip_idx, _ = max(tips, key=lambda item: item[1].end_pos.y)
            elif movement == 3: # Left
                selected_tip_idx, _ = min(tips, key=lambda item: item[1].end_pos.x)
            elif movement == 4: # Right
                selected_tip_idx, _ = max(tips, key=lambda item: item[1].end_pos.x)
        except ValueError:
            return False

        if selected_tip_idx == -1:
            return False

        parent_branch = self.branches[selected_tip_idx]
        parent_branch.is_tip = False
        self.last_action_tip_idx = selected_tip_idx
        self.last_action_feedback_timer = 15 # frames

        available_resources = sorted(self.resources, key=lambda r: parent_branch.end_pos.distance_to(r))
        targets = available_resources[:2]

        if not targets:
            return True 

        for target_pos in targets:
            target_vec = (target_pos - parent_branch.end_pos).normalize()
            new_angle = pygame.Vector2(0, -1).angle_to(target_vec)
            new_length = parent_branch.max_length * self.np_random.uniform(0.7, 0.9)
            new_width = parent_branch.width * 0.8
            new_gen = parent_branch.generation + 1
            self.branches.append(Branch(parent_branch.end_pos, new_angle, new_length, new_width, new_gen, 2.0, target_pos))
        
        return True

    def _update_world_state(self):
        """Updates branches, particles, and checks for resource collection."""
        shaping_reward = 0.0
        collection_reward = 0.0
        
        for branch in self.branches:
            if branch.is_growing:
                dist_change = branch.update()
                shaping_reward += max(0, dist_change) * 0.1 # Per-step reward for getting closer

        collected_indices = set()
        growing_tips = [b for b in self.branches if b.is_growing and b.is_tip]
        for branch in growing_tips:
            for i, res_pos in enumerate(self.resources):
                if i not in collected_indices and branch.end_pos.distance_to(res_pos) < self.RESOURCE_COLLECT_RADIUS:
                    collected_indices.add(i)
                    self.score += 1
                    collection_reward += 5.0 # Event-based reward for collection
                    branch.is_growing = False 
                    self._spawn_particles(res_pos)
                    # sfx: resource_collect.play()

        if collected_indices:
            self.resources = [res for i, res in enumerate(self.resources) if i not in collected_indices]

        self.particles = [p for p in self.particles if p.lifetime > 0]
        for p in self.particles:
            p.update()
            
        return shaping_reward, collection_reward

    def _spawn_particles(self, pos):
        """Creates a burst of particles for a satisfying collection effect."""
        for _ in range(15):
            angle = self.np_random.uniform(0, 360)
            speed = self.np_random.uniform(1, 4)
            velocity = pygame.Vector2(speed, 0).rotate(angle)
            lifetime = self.np_random.integers(20, 40)
            radius = self.np_random.uniform(2, 5)
            color = random.choice([(50, 180, 255), (150, 220, 255), (200, 240, 255)])
            self.particles.append(Particle(pos, color, lifetime, radius, velocity))

    def _get_observation(self):
        self.screen.blit(self.background_surface, (0, 0))
        self._draw_resources()
        self._draw_tree()
        self._draw_particles()
        self._draw_action_feedback()
        self._draw_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _get_info(self):
        return {"score": self.score, "steps": self.steps}

    def _create_gradient_background(self):
        """Creates a surface with a vertical color gradient for a polished look."""
        bg = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        for y in range(self.SCREEN_HEIGHT):
            ratio = y / self.SCREEN_HEIGHT
            r = int(self.COLOR_BG_START[0] * (1 - ratio) + self.COLOR_BG_END[0] * ratio)
            g = int(self.COLOR_BG_START[1] * (1 - ratio) + self.COLOR_BG_END[1] * ratio)
            b = int(self.COLOR_BG_START[2] * (1 - ratio) + self.COLOR_BG_END[2] * ratio)
            pygame.draw.line(bg, (r, g, b), (0, y), (self.SCREEN_WIDTH, y))
        return bg

    def _draw_tree(self):
        for branch in self.branches:
            branch.draw(self.screen)

    def _draw_resources(self):
        for pos in self.resources:
            x, y = int(pos.x), int(pos.y)
            pygame.gfxdraw.filled_circle(self.screen, x, y, self.RESOURCE_RADIUS, self.COLOR_RESOURCE)
            pygame.gfxdraw.aacircle(self.screen, x, y, self.RESOURCE_RADIUS, self.COLOR_RESOURCE_OUTLINE)

    def _draw_particles(self):
        for p in self.particles:
            p.draw(self.screen)
    
    def _draw_action_feedback(self):
        """Draws a highlight on the branch that was just split to provide clear feedback."""
        if self.last_action_feedback_timer > 0 and 0 <= self.last_action_tip_idx < len(self.branches):
            branch = self.branches[self.last_action_tip_idx]
            radius = 20 * (self.last_action_feedback_timer / 15) # Shrinks
            alpha = int(255 * (self.last_action_feedback_timer / 15)) # Fades out
            
            s = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            color = self.COLOR_SELECTOR + (alpha,)
            pygame.gfxdraw.aacircle(s, int(branch.end_pos.x), int(branch.end_pos.y), int(radius), color)
            self.screen.blit(s, (0, 0))

    def _draw_ui(self):
        score_text = self.font_ui.render(f"SCORE: {self.score}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (10, 10))
        
        time_left = (self.MAX_STEPS - self.steps) / self.FPS
        timer_color = self.COLOR_UI_TEXT if time_left > 10 else self.COLOR_TIMER_WARN
        timer_text = self.font_timer.render(f"{time_left:.1f}s", True, timer_color)
        self.screen.blit(timer_text, (self.SCREEN_WIDTH - timer_text.get_width() - 10, 10))

    def close(self):
        pygame.quit()


if __name__ == "__main__":
    # This block allows you to run the environment directly for testing.
    # It's not part of the Gymnasium interface but is useful for development.
    # To use, you might need to unset the dummy video driver:
    # `os.environ.pop("SDL_VIDEODRIVER", None)` before `pygame.display.set_mode`
    
    # We keep the dummy driver for headless testing, but a visible window won't be created.
    # The logic runs and we can print outputs. For visual debugging, comment out the os.environ line at the top.
    
    env = GameEnv()
    obs, info = env.reset()
    
    # To run with a visible window, you would need a display.
    # For this example, we'll just simulate steps and print info.
    print(f"Game Description: {GameEnv.game_description}")
    print(f"User Guide: {GameEnv.user_guide}")
    print("Running a test episode...")
    
    terminated = False
    total_reward = 0
    step_count = 0
    
    while not terminated:
        action = env.action_space.sample() # Take a random action
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        step_count += 1
        
        if terminated or truncated:
            print(f"Episode finished after {step_count} steps.")
            print(f"Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
            break
            
    env.close()
    print("Test episode complete.")