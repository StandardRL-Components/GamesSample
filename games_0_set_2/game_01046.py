
# Generated: 2025-08-27T15:41:05.991786
# Source Brief: brief_01046.md
# Brief Index: 1046

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
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

    user_guide = (
        "Controls: ←→ to select a target asteroid. Press space to hop. Collect yellow fuel cells."
    )

    game_description = (
        "Navigate a treacherous asteroid field, hopping between rocks and collecting fuel to reach the green end zone."
    )

    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Game constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.MAX_ASTEROIDS = 30
        self.MIN_ASTEROID_RADIUS = 15
        self.MAX_ASTEROID_RADIUS = 35
        self.STARTING_FUEL = 10
        self.FUEL_PER_CELL = 5
        self.HOP_COST = 1
        self.MAX_HOP_DISTANCE = 200
        self.END_ZONE_WIDTH = 50
        self.VICTORY_FUEL_REQ = 5
        self.MAX_TIME_STEPS = 100 # Number of hops allowed

        # Colors
        self.COLOR_BG = (10, 15, 30)
        self.COLOR_STAR = (200, 200, 220)
        self.COLOR_PLAYER = (60, 180, 255)
        self.COLOR_PLAYER_GLOW = (60, 180, 255, 50)
        self.COLOR_ASTEROID = (100, 100, 110)
        self.COLOR_FUEL = (255, 220, 0)
        self.COLOR_END_ZONE = (0, 200, 100)
        self.COLOR_UI_TEXT = (255, 255, 255)
        self.COLOR_SELECTOR = (255, 50, 50)
        self.COLOR_SELECTOR_LINE = (255, 255, 255, 100)
        
        # Gymnasium spaces
        self.observation_space = Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("monospace", 20, bold=True)
        self.font_msg = pygame.font.SysFont("monospace", 40, bold=True)

        # State variables (initialized in reset)
        self.asteroids = []
        self.stars = []
        self.player_asteroid_idx = 0
        self.selected_target_idx = -1
        self.reachable_targets = []
        self.fuel = 0
        self.score = 0
        self.steps = 0
        self.time_left = 0
        self.game_over = False
        self.game_outcome = ""
        self.end_zone_rect = None
        self.hop_animation = None

        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.score = 0
        self.steps = 0
        self.game_over = False
        self.game_outcome = ""
        self.hop_animation = None
        
        self._setup_level()
        
        return self._get_observation(), self._get_info()

    def _setup_level(self):
        """Initializes a new level, regenerating asteroids and player state."""
        self.fuel = self.STARTING_FUEL
        self.time_left = self.MAX_TIME_STEPS
        
        # Generate stars for parallax effect
        self.stars = [
            (
                (self.np_random.integers(0, self.WIDTH), self.np_random.integers(0, self.HEIGHT)),
                self.np_random.integers(1, 3)
            )
            for _ in range(100)
        ]

        # Generate asteroids ensuring no major overlaps
        self._generate_asteroids()

        # Place player and end zone
        self._place_player_and_endzone()

        # Assign fuel to some asteroids
        self._assign_fuel()

        # Find initial reachable targets
        self._update_reachable_targets()
        
        # Anti-softlock: if no targets are reachable, reset the level
        if not self.reachable_targets:
            self._setup_level()

    def _generate_asteroids(self):
        self.asteroids = []
        for _ in range(self.MAX_ASTEROIDS):
            placed = False
            while not placed:
                radius = self.np_random.integers(self.MIN_ASTEROID_RADIUS, self.MAX_ASTEROID_RADIUS + 1)
                pos = pygame.Vector2(
                    self.np_random.integers(radius, self.WIDTH - self.END_ZONE_WIDTH - radius),
                    self.np_random.integers(radius, self.HEIGHT - radius)
                )
                
                # Check for overlap with existing asteroids
                overlap = False
                for other in self.asteroids:
                    if pos.distance_to(other['pos']) < radius + other['radius'] + 10:
                        overlap = True
                        break
                if not overlap:
                    self.asteroids.append({
                        'pos': pos,
                        'radius': radius,
                        'has_fuel': False,
                        'color': (
                            self.np_random.integers(80, 121),
                            self.np_random.integers(80, 121),
                            self.np_random.integers(90, 131)
                        )
                    })
                    placed = True

    def _place_player_and_endzone(self):
        # Sort asteroids by x-position to find the leftmost one for the player
        self.asteroids.sort(key=lambda a: a['pos'].x)
        self.player_asteroid_idx = 0

        # Place end zone on the right side
        self.end_zone_rect = pygame.Rect(
            self.WIDTH - self.END_ZONE_WIDTH, 0, self.END_ZONE_WIDTH, self.HEIGHT
        )

    def _assign_fuel(self):
        # Assign fuel to a random subset of asteroids (not the player's starting one)
        num_fuel_cells = self.np_random.integers(5, 10)
        fuel_candidates = list(range(1, len(self.asteroids)))
        self.np_random.shuffle(fuel_candidates)
        for i in range(min(num_fuel_cells, len(fuel_candidates))):
            self.asteroids[fuel_candidates[i]]['has_fuel'] = True

    def _update_reachable_targets(self):
        player_pos = self.asteroids[self.player_asteroid_idx]['pos']
        
        # Find all asteroids within hop distance, excluding the current one
        potential_targets = []
        for i, asteroid in enumerate(self.asteroids):
            if i != self.player_asteroid_idx:
                dist = player_pos.distance_to(asteroid['pos'])
                if dist <= self.MAX_HOP_DISTANCE:
                    angle = math.atan2(asteroid['pos'].y - player_pos.y, asteroid['pos'].x - player_pos.x)
                    potential_targets.append({'index': i, 'angle': angle, 'dist': dist})
        
        # Sort targets by angle for consistent cycling
        potential_targets.sort(key=lambda t: t['angle'])
        self.reachable_targets = [t['index'] for t in potential_targets]
        
        # Set a new default selection
        if self.reachable_targets:
            if self.selected_target_idx not in self.reachable_targets:
                self.selected_target_idx = self.reachable_targets[0]
        else:
            self.selected_target_idx = -1

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement, space_pressed, _ = action
        reward = 0
        terminated = False
        self.steps += 1
        self.time_left -= 1

        # 1. Handle target selection (arrow keys)
        if movement in [3, 4] and self.reachable_targets: # 3=left, 4=right
            try:
                current_selection_index = self.reachable_targets.index(self.selected_target_idx)
                if movement == 3: # Left
                    current_selection_index = (current_selection_index - 1) % len(self.reachable_targets)
                elif movement == 4: # Right
                    current_selection_index = (current_selection_index + 1) % len(self.reachable_targets)
                self.selected_target_idx = self.reachable_targets[current_selection_index]
            except ValueError: # If current selection is somehow invalid, reset it
                self.selected_target_idx = self.reachable_targets[0]

        # 2. Handle hop execution (space bar)
        if space_pressed and self.selected_target_idx != -1:
            # Sfx: Hop_Charge.wav
            reward, terminated = self._execute_hop()
            
        # 3. Check for termination from time
        if self.time_left <= 0 and not terminated:
            self.game_over = True
            terminated = True
            reward -= 10
            self.game_outcome = "OUT OF TIME"

        if terminated:
            self.game_over = True

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _execute_hop(self):
        reward = 0
        terminated = False

        prev_pos = self.asteroids[self.player_asteroid_idx]['pos']
        dist_to_end_before = self.WIDTH - prev_pos.x

        # Update player position
        self.player_asteroid_idx = self.selected_target_idx
        new_pos = self.asteroids[self.player_asteroid_idx]['pos']
        dist_to_end_after = self.WIDTH - new_pos.x

        # Set up animation
        self.hop_animation = {'start': prev_pos, 'end': new_pos, 'progress': 0.0}

        # Apply fuel cost
        self.fuel -= self.HOP_COST
        
        # Reward for progress
        if dist_to_end_after < dist_to_end_before:
            reward += 0.1 # Closer to goal
        else:
            reward -= 0.2 # Further from goal

        # Check new asteroid for fuel
        current_asteroid = self.asteroids[self.player_asteroid_idx]
        if current_asteroid['has_fuel']:
            # Sfx: Fuel_Collect.wav
            self.fuel += self.FUEL_PER_CELL
            current_asteroid['has_fuel'] = False
            reward += 1.0
            self.score += 10

        # Check for termination from fuel loss
        if self.fuel <= 0:
            terminated = True
            reward -= 10
            self.game_outcome = "OUT OF FUEL"

        # Check for victory condition
        if self.end_zone_rect.collidepoint(new_pos):
            if self.fuel >= self.VICTORY_FUEL_REQ:
                # Sfx: Victory.wav
                reward += 100
                self.score += 100
                self.game_outcome = "SUCCESS!"
            else:
                # Sfx: Failure.wav
                reward -= 10
                self.game_outcome = "NOT ENOUGH FUEL"
            terminated = True

        # Update targets for the next turn
        self._update_reachable_targets()
        
        # If no more moves are possible
        if not self.reachable_targets and not terminated:
            terminated = True
            reward -= 10
            self.game_outcome = "STRANDED!"

        return reward, terminated

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Render stars
        for pos, radius in self.stars:
            pygame.draw.circle(self.screen, self.COLOR_STAR, pos, radius)

        # Render end zone
        end_zone_surface = pygame.Surface(self.end_zone_rect.size, pygame.SRCALPHA)
        end_zone_surface.fill((*self.COLOR_END_ZONE, 50))
        self.screen.blit(end_zone_surface, self.end_zone_rect.topleft)

        # Render hop indicators
        if not self.game_over and self.reachable_targets:
            player_pos = self.asteroids[self.player_asteroid_idx]['pos']
            for target_idx in self.reachable_targets:
                target_pos = self.asteroids[target_idx]['pos']
                if target_idx == self.selected_target_idx:
                    pygame.gfxdraw.line(self.screen, int(player_pos.x), int(player_pos.y), int(target_pos.x), int(target_pos.y), self.COLOR_SELECTOR_LINE)
                    pygame.gfxdraw.aacircle(self.screen, int(target_pos.x), int(target_pos.y), self.asteroids[target_idx]['radius'] + 5, self.COLOR_SELECTOR)
                else:
                    pygame.gfxdraw.line(self.screen, int(player_pos.x), int(player_pos.y), int(target_pos.x), int(target_pos.y), (*self.COLOR_STAR, 50))

        # Render asteroids and fuel
        for asteroid in self.asteroids:
            pygame.gfxdraw.filled_circle(self.screen, int(asteroid['pos'].x), int(asteroid['pos'].y), asteroid['radius'], asteroid['color'])
            pygame.gfxdraw.aacircle(self.screen, int(asteroid['pos'].x), int(asteroid['pos'].y), asteroid['radius'], tuple(c*0.8 for c in asteroid['color']))
            if asteroid['has_fuel']:
                fuel_size = int(asteroid['radius'] * 0.6)
                pygame.draw.rect(self.screen, self.COLOR_FUEL, (asteroid['pos'].x - fuel_size/2, asteroid['pos'].y - fuel_size/2, fuel_size, fuel_size))

        # Render player
        player_pos = self.asteroids[self.player_asteroid_idx]['pos']
        player_radius = 12

        # Animate hop
        if self.hop_animation:
            self.hop_animation['progress'] = min(1.0, self.hop_animation['progress'] + 0.25)
            player_pos = self.hop_animation['start'].lerp(self.hop_animation['end'], self.hop_animation['progress'])
            if self.hop_animation['progress'] >= 1.0:
                self.hop_animation = None

        # Player glow
        pygame.gfxdraw.filled_circle(self.screen, int(player_pos.x), int(player_pos.y), player_radius + 5, self.COLOR_PLAYER_GLOW)

        # Player triangle
        p1 = player_pos + pygame.Vector2(0, -player_radius).rotate(0)
        p2 = player_pos + pygame.Vector2(0, -player_radius).rotate(135)
        p3 = player_pos + pygame.Vector2(0, -player_radius).rotate(-135)
        
        # Point towards selected target
        if self.selected_target_idx != -1 and not self.game_over:
            target_pos = self.asteroids[self.selected_target_idx]['pos']
            angle = math.degrees(math.atan2(player_pos.y - target_pos.y, target_pos.x - player_pos.x))
            p1 = player_pos + pygame.Vector2(0, -player_radius).rotate(-angle-90)
            p2 = player_pos + pygame.Vector2(0, -player_radius).rotate(-angle+45)
            p3 = player_pos + pygame.Vector2(0, -player_radius).rotate(-angle-225)
        
        pygame.gfxdraw.aapolygon(self.screen, [(p1.x, p1.y), (p2.x, p2.y), (p3.x, p3.y)], self.COLOR_PLAYER)
        pygame.gfxdraw.filled_polygon(self.screen, [(p1.x, p1.y), (p2.x, p2.y), (p3.x, p3.y)], self.COLOR_PLAYER)

    def _render_ui(self):
        # Fuel display
        fuel_text = self.font_ui.render(f"FUEL: {self.fuel}", True, self.COLOR_UI_TEXT)
        self.screen.blit(fuel_text, (10, 10))

        # Time display
        time_text = self.font_ui.render(f"HOPS LEFT: {self.time_left}", True, self.COLOR_UI_TEXT)
        self.screen.blit(time_text, (self.WIDTH - time_text.get_width() - 10, 10))

        # Score display
        score_text = self.font_ui.render(f"SCORE: {self.score}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (10, 35))

        # Game Over message
        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            self.screen.blit(overlay, (0, 0))
            
            msg_text = self.font_msg.render(self.game_outcome, True, self.COLOR_UI_TEXT)
            msg_rect = msg_text.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
            self.screen.blit(msg_text, msg_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "fuel": self.fuel,
            "time_left": self.time_left,
        }

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        """Call this at the end of __init__ to verify implementation."""
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
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == "__main__":
    env = GameEnv(render_mode="rgb_array")
    
    # --- Manual Play ---
    # This allows a human to play the game.
    obs, info = env.reset()
    done = False
    
    # Use a Pygame window to display the game.
    pygame.display.set_caption(env.game_description)
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    
    action = env.action_space.sample()
    action.fill(0) # Start with no-op

    while not done:
        # Convert numpy array to pygame surface for display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        # Reset actions at the start of each frame
        action.fill(0)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
            
            # This is a turn-based game, so we only process one key press per frame
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    action[0] = 3
                elif event.key == pygame.K_RIGHT:
                    action[0] = 4
                elif event.key == pygame.K_SPACE:
                    action[1] = 1
                elif event.key == pygame.K_r: # Reset on 'r' key
                    obs, info = env.reset()
                    continue
                
                # Take a step in the environment with the chosen action
                obs, reward, terminated, truncated, info = env.step(action)
                print(f"Action: {action}, Reward: {reward:.2f}, Info: {info}")
                
                if terminated or truncated:
                    print("Game Over. Press 'r' to restart.")

    env.close()