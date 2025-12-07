import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T14:06:48.902398
# Source Brief: brief_00215.md
# Brief Index: 215
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

# Helper class for Ant entity
class Ant:
    """Represents a single ant in the colony."""
    def __init__(self, x, y, world_scale):
        self.pos = np.array([float(x), float(y)])
        self.visual_pos = np.array([float(x), float(y)])
        self.world_scale = world_scale
        self.size = 5  # Logical radius
        self.speed = 1.0
        self.has_food = False
        self.color = (200, 50, 50)
        self.lerp_factor = 0.3 # For smooth visual movement

    def move(self, dx, dy, bounds):
        """Updates the logical position of the ant."""
        new_pos = self.pos + np.array([dx, dy]) * self.speed
        self.pos[0] = np.clip(new_pos[0], self.size, bounds[0] - self.size)
        self.pos[1] = np.clip(new_pos[1], self.size, bounds[1] - self.size)
        
    def update_visual_pos(self):
        """Interpolates visual position towards logical position for smooth animation."""
        self.visual_pos = (1 - self.lerp_factor) * self.visual_pos + self.lerp_factor * self.pos

    def merge_with(self, other_ant):
        """Absorbs another ant, growing in size and speed."""
        # sound placeholder: # sfx_merge
        self.size = min(15, self.size + other_ant.size / 2) # Cap size
        self.speed = min(2.0, self.speed + 0.1) # Cap speed

    def draw(self, surface, is_selected):
        """Renders the ant on the Pygame surface."""
        screen_pos = (int(self.visual_pos[0] * self.world_scale), int(self.visual_pos[1] * self.world_scale))
        screen_radius = int(self.size * self.world_scale)

        if is_selected:
            # Draw a glowing highlight for the selected ant
            highlight_radius = int(screen_radius * 1.5)
            highlight_color = (255, 255, 255)
            pygame.gfxdraw.filled_circle(surface, screen_pos[0], screen_pos[1], highlight_radius, (*highlight_color, 40))
            pygame.gfxdraw.aacircle(surface, screen_pos[0], screen_pos[1], highlight_radius, (*highlight_color, 80))
            pygame.gfxdraw.aacircle(surface, screen_pos[0], screen_pos[1], screen_radius, (255, 255, 255))

        # Draw ant body
        pygame.gfxdraw.filled_circle(surface, screen_pos[0], screen_pos[1], screen_radius, self.color)
        pygame.gfxdraw.aacircle(surface, screen_pos[0], screen_pos[1], screen_radius, (0,0,0))
        
        # Draw food pellet if carrying one
        if self.has_food:
            food_color = (255, 220, 0)
            food_size = int(self.world_scale * 2)
            food_rect = pygame.Rect(screen_pos[0] - food_size // 2, screen_pos[1] - screen_radius - food_size, food_size, food_size)
            pygame.draw.rect(surface, food_color, food_rect)
            pygame.draw.rect(surface, (0,0,0), food_rect, 1)


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}
    
    game_description = (
        "Control an ant colony to gather food and return it to the anthill. "
        "Merge ants to create stronger units before time runs out."
    )
    user_guide = (
        "Controls: Use the arrow keys (↑↓←→) to move the selected ant. "
        "Press TAB to cycle between ants."
    )
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # --- Constants ---
        self.SCREEN_WIDTH, self.SCREEN_HEIGHT = 640, 400
        self.WORLD_WIDTH, self.WORLD_HEIGHT = 64, 40
        self.SCALE = self.SCREEN_WIDTH / self.WORLD_WIDTH
        self.FPS = 30
        self.MAX_STEPS = self.FPS * 60  # 60 seconds
        self.WIN_SCORE = 10
        self.INITIAL_ANTS = 5
        self.INITIAL_FOOD = 20
        self.MERGE_DISTANCE = 2.0

        # Colors
        self.COLOR_BG = (51, 38, 28)
        self.COLOR_ANTHILL = (100, 180, 80)
        self.COLOR_FOOD = (255, 220, 0)
        self.COLOR_UI_TEXT = (230, 230, 230)

        # EXACT spaces:
        self.observation_space = Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.Font(None, 48)
        self.font_small = pygame.font.Font(None, 32)
        
        # Initialize state variables
        self.ants = []
        self.food_pellets = []
        self.anthill_pos = np.array([self.WORLD_WIDTH / 2, self.WORLD_HEIGHT / 2])
        self.anthill_radius = 4.0
        self.steps = 0
        self.score = 0
        self.selected_ant_idx = 0
        
        self.rng = np.random.default_rng()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
            random.seed(seed)
        
        # Initialize all game state
        self.steps = 0
        self.score = 0
        
        self._spawn_ants()
        self._spawn_food()
        
        self.selected_ant_idx = 0
        
        # MUST return exactly this tuple
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        reward = 0.0
        terminated = False
        
        # Unpack factorized action
        movement = action[0]  # 0-4: none/cycle/up/down/left/right
        
        # --- Action Handling ---
        if len(self.ants) > 0:
            if movement == 0: # Cycle selected ant
                self.selected_ant_idx = (self.selected_ant_idx + 1) % len(self.ants)
            else: # Move selected ant
                ant = self.ants[self.selected_ant_idx]
                
                # Calculate reward-relevant distances before moving
                old_pos = ant.pos.copy()
                old_dist_to_anthill = np.linalg.norm(old_pos - self.anthill_pos)
                old_dist_to_food = self._get_dist_to_nearest_food(old_pos)

                # Move ant
                move_map = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}
                dx, dy = move_map[movement]
                ant.move(dx, dy, (self.WORLD_WIDTH, self.WORLD_HEIGHT))

                # Calculate reward-relevant distances after moving
                new_dist_to_anthill = np.linalg.norm(ant.pos - self.anthill_pos)
                new_dist_to_food = self._get_dist_to_nearest_food(ant.pos)
                
                # Distance-based rewards
                if ant.has_food:
                    reward += 0.1 * (old_dist_to_anthill - new_dist_to_anthill)
                elif old_dist_to_food is not None and new_dist_to_food is not None:
                    # Reward for getting closer to food
                    reward += 0.05 * (old_dist_to_food - new_dist_to_food)

        # --- Game Logic ---
        reward += self._handle_interactions()
        reward += self._handle_merging()

        self.steps += 1
        
        # --- Termination Check ---
        if self.score >= self.WIN_SCORE:
            terminated = True
            reward += 100  # Win bonus
        elif self.steps >= self.MAX_STEPS:
            terminated = True
            reward -= 100 # Loss penalty
        elif len(self.ants) == 0:
            terminated = True
            reward -= 100 # Loss penalty

        # MUST return exactly this 5-tuple
        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )

    def _spawn_ants(self):
        self.ants = []
        for _ in range(self.INITIAL_ANTS):
            # Spawn near the anthill
            angle = random.uniform(0, 2 * math.pi)
            dist = self.anthill_radius + 1.0
            x = self.anthill_pos[0] + math.cos(angle) * dist
            y = self.anthill_pos[1] + math.sin(angle) * dist
            self.ants.append(Ant(x, y, self.SCALE))
    
    def _spawn_food(self):
        self.food_pellets = []
        for _ in range(self.INITIAL_FOOD):
            # Avoid spawning on the anthill
            while True:
                pos = np.array([
                    random.uniform(2, self.WORLD_WIDTH - 2),
                    random.uniform(2, self.WORLD_HEIGHT - 2)
                ])
                if np.linalg.norm(pos - self.anthill_pos) > self.anthill_radius + 5.0:
                    self.food_pellets.append(pos)
                    break

    def _get_dist_to_nearest_food(self, pos):
        if not self.food_pellets:
            return None
        distances = [np.linalg.norm(pos - food_pos) for food_pos in self.food_pellets]
        return min(distances)

    def _handle_interactions(self):
        """Check for food pickup and delivery for the selected ant."""
        if len(self.ants) == 0:
            return 0
        
        reward = 0
        ant = self.ants[self.selected_ant_idx]

        # Food pickup
        if not ant.has_food:
            for i, food_pos in reversed(list(enumerate(self.food_pellets))):
                if np.linalg.norm(ant.pos - food_pos) < ant.size:
                    ant.has_food = True
                    del self.food_pellets[i]
                    reward += 1.0  # Pickup reward
                    # sound placeholder: # sfx_pickup
                    break # Can only pick up one
        
        # Food delivery
        if ant.has_food:
            if np.linalg.norm(ant.pos - self.anthill_pos) < ant.size + self.anthill_radius:
                ant.has_food = False
                self.score += 1
                reward += 5.0 # Delivery reward
                # sound placeholder: # sfx_delivery

        return reward

    def _handle_merging(self):
        """Check for and handle ant merging."""
        reward = 0
        i = len(self.ants) - 1
        while i > 0:
            j = i - 1
            merged = False
            while j >= 0:
                ant1 = self.ants[i]
                ant2 = self.ants[j]
                if np.linalg.norm(ant1.pos - ant2.pos) < (ant1.size + ant2.size) / 2 * 0.5: # Use a fraction of combined size
                    # Ant j absorbs ant i
                    ant2.merge_with(ant1)
                    del self.ants[i]
                    reward += 2.0

                    # Adjust selected index if necessary
                    if self.selected_ant_idx == i:
                        self.selected_ant_idx = j
                    elif self.selected_ant_idx > i:
                        self.selected_ant_idx -= 1
                    merged = True
                    break
                j -= 1
            if merged:
                i -= 1
                continue
            i -= 1
        
        # Ensure selected index is valid after potential merges
        if len(self.ants) > 0:
            self.selected_ant_idx = self.selected_ant_idx % len(self.ants)
        
        return reward

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
        # Draw anthill
        ah_pos = (int(self.anthill_pos[0] * self.SCALE), int(self.anthill_pos[1] * self.SCALE))
        ah_radius = int(self.anthill_radius * self.SCALE)
        pygame.gfxdraw.filled_circle(self.screen, ah_pos[0], ah_pos[1], ah_radius, self.COLOR_ANTHILL)
        pygame.gfxdraw.aacircle(self.screen, ah_pos[0], ah_pos[1], ah_radius, (0,0,0))
        
        # Draw food pellets
        food_size = int(2 * self.SCALE)
        for food_pos in self.food_pellets:
            screen_pos = (int(food_pos[0] * self.SCALE), int(food_pos[1] * self.SCALE))
            food_rect = pygame.Rect(screen_pos[0] - food_size//2, screen_pos[1] - food_size//2, food_size, food_size)
            pygame.draw.rect(self.screen, self.COLOR_FOOD, food_rect)
            pygame.draw.rect(self.screen, (0,0,0), food_rect, 1)

        # Draw ants
        for i, ant in enumerate(self.ants):
            ant.update_visual_pos()
            ant.draw(self.screen, i == self.selected_ant_idx)

    def _render_ui(self):
        # Render score
        score_text = self.font_large.render(f"{self.score} / {self.WIN_SCORE}", True, self.COLOR_UI_TEXT)
        score_pos_world = self.anthill_pos + np.array([0, self.anthill_radius + 2])
        score_pos_screen = (int(score_pos_world[0] * self.SCALE - score_text.get_width() / 2),
                            int(score_pos_world[1] * self.SCALE))
        self.screen.blit(score_text, score_pos_screen)

        # Render timer
        time_left = (self.MAX_STEPS - self.steps) / self.FPS
        timer_text = self.font_small.render(f"Time: {max(0, time_left):.1f}", True, self.COLOR_UI_TEXT)
        self.screen.blit(timer_text, (self.SCREEN_WIDTH - timer_text.get_width() - 10, 10))
        
        # Render selected ant info (optional but helpful)
        if len(self.ants) > 0:
            ant_info_text = self.font_small.render(f"Ant: {self.selected_ant_idx + 1}/{len(self.ants)}", True, self.COLOR_UI_TEXT)
            self.screen.blit(ant_info_text, (10, 10))

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "ants_remaining": len(self.ants),
            "time_left_seconds": (self.MAX_STEPS - self.steps) / self.FPS,
        }
    
    def close(self):
        pygame.quit()

# Example usage:
if __name__ == '__main__':
    # Set a non-dummy driver for manual play
    os.environ["SDL_VIDEODRIVER"] = "x11" 
    
    env = GameEnv()
    obs, info = env.reset(seed=42)
    
    # --- Manual Play Loop ---
    running = True
    terminated = False
    
    # Use a dictionary to track held keys for fluid movement
    keys_held = {
        pygame.K_UP: False,
        pygame.K_DOWN: False,
        pygame.K_LEFT: False,
        pygame.K_RIGHT: False,
        pygame.K_LSHIFT: False,
        pygame.K_SPACE: False,
        pygame.K_TAB: False, # For cycling ants
    }
    
    # Re-create screen for display
    display_screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    pygame.display.set_caption("Ant Colony Gym Environment")
    
    while running:
        action = [0, 0, 0] # Default action: [no-op, space_up, shift_up]
        
        # Event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key in keys_held:
                    keys_held[event.key] = True
                if event.key == pygame.K_r: # Reset on 'r'
                    obs, info = env.reset(seed=42)
                    terminated = False
            if event.type == pygame.KEYUP:
                if event.key in keys_held:
                    keys_held[event.key] = False

        # Map keyboard state to action space
        # Note: The logic only uses action[0], but we populate all for completeness
        move_action = 0
        if keys_held[pygame.K_UP]:
            move_action = 1
        elif keys_held[pygame.K_DOWN]:
            move_action = 2
        elif keys_held[pygame.K_LEFT]:
            move_action = 3
        elif keys_held[pygame.K_RIGHT]:
            move_action = 4
        
        # TAB press to cycle ants (maps to movement=0)
        if keys_held[pygame.K_TAB]:
            move_action = 0
            # Make it a single press, not hold, by consuming the key press
            keys_held[pygame.K_TAB] = False 
            
        action[0] = move_action
        action[1] = 1 if keys_held[pygame.K_SPACE] else 0
        action[2] = 1 if keys_held[pygame.K_LSHIFT] else 0
        
        if not terminated:
            # We only need to step if an action is taken or auto_advance is on
            if env.auto_advance or any(k for k,v in keys_held.items() if v) or move_action != 0:
                 obs, reward, terminated, truncated, info = env.step(action)
            
            if terminated:
                print(f"Episode Finished. Final Info: {info}")

        # Render the observation to the display window
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        display_screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        env.clock.tick(env.FPS)

    env.close()