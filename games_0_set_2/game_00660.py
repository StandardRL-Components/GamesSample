
# Generated: 2025-08-27T14:22:13.077015
# Source Brief: brief_00660.md
# Brief Index: 660

        
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
        "Controls: Use arrow keys to select a target wormhole and travel to it. "
        "Reach the green exit portal before time runs out!"
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Navigate a spaceship through a chaotic field of wormholes and "
        "obstacles to reach the exit within the time limit."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    FPS = 30
    GAME_DURATION_SECONDS = 60

    COLOR_BG = (10, 0, 20)
    COLOR_PLAYER = (255, 255, 255)
    COLOR_OBSTACLE = (255, 50, 50)
    COLOR_EXIT = (50, 255, 50)
    COLOR_TEXT = (240, 240, 240)
    WORMHOLE_COLORS = [
        (0, 150, 255), (255, 150, 0), (200, 0, 255),
        (0, 255, 150), (255, 255, 0)
    ]

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
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
        self.player_pos = pygame.Vector2(0, 0)
        self.player_angle = 0
        self.current_wormhole_idx = 0
        self.wormholes = []
        self.obstacles = []
        self.exit_rect = pygame.Rect(0, 0, 0, 0)
        self.particles = []
        self.steps = 0
        self.score = 0.0
        self.game_over = False
        self.time_remaining = 0
        self.rng = np.random.default_rng()

        self.reset()
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self.steps = 0
        self.score = 0.0
        self.game_over = False
        self.time_remaining = self.GAME_DURATION_SECONDS * self.FPS
        self.particles = []
        
        self._generate_level()
        
        return self._get_observation(), self._get_info()

    def _generate_level(self):
        # Place exit portal on the right side
        exit_w, exit_h = 40, 80
        self.exit_rect = pygame.Rect(
            self.SCREEN_WIDTH - exit_w - 20,
            self.rng.integers(20, self.SCREEN_HEIGHT - exit_h - 20),
            exit_w, exit_h
        )

        # Generate wormholes
        num_wormholes = self.rng.integers(25, 40)
        self.wormholes = []
        for _ in range(num_wormholes):
            pos = pygame.Vector2(
                self.rng.integers(50, self.SCREEN_WIDTH - 50),
                self.rng.integers(50, self.SCREEN_HEIGHT - 50)
            )
            self.wormholes.append({
                "pos": pos,
                "radius": self.rng.integers(10, 15),
                "color": random.choice(self.WORMHOLE_COLORS),
                "pulse_offset": self.rng.random() * 2 * math.pi
            })

        # Set player start position at a wormhole on the left
        leftmost_wormholes = sorted([
            (i, wh) for i, wh in enumerate(self.wormholes)
        ], key=lambda item: item[1]['pos'].x)
        
        self.current_wormhole_idx = leftmost_wormholes[0][0]
        self.player_pos = self.wormholes[self.current_wormhole_idx]['pos'].copy()
        self.player_angle = 0

        # Generate obstacles, avoiding player start and exit
        num_obstacles = self.rng.integers(15, 25)
        self.obstacles = []
        for _ in range(num_obstacles):
            size = self.rng.integers(20, 35)
            while True:
                pos = pygame.Vector2(
                    self.rng.integers(0, self.SCREEN_WIDTH - size),
                    self.rng.integers(0, self.SCREEN_HEIGHT - size)
                )
                obstacle_rect = pygame.Rect(pos.x, pos.y, size, size)
                
                # Check for collisions with start/exit
                if obstacle_rect.colliderect(self.exit_rect.inflate(40, 40)):
                    continue
                if obstacle_rect.clipline(self.player_pos, self.player_pos):
                     continue
                if pos.distance_to(self.player_pos) < 100:
                    continue
                
                self.obstacles.append({
                    "rect": obstacle_rect,
                    "angle": self.rng.random() * 360,
                    "speed": self.rng.uniform(-2, 2)
                })
                break

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        movement = action[0]
        reward = -0.01  # Time penalty

        prev_dist_to_exit = self.player_pos.distance_to(self.exit_rect.center)
        
        # Player is always at a wormhole, action selects the next one
        if movement != 0:
            target_idx = self._find_target_wormhole(movement)
            if target_idx is not None and target_idx != self.current_wormhole_idx:
                # # Sound effect placeholder: player warp
                old_pos = self.player_pos.copy()
                self.current_wormhole_idx = target_idx
                self.player_pos = self.wormholes[target_idx]['pos'].copy()
                
                # Update player angle to face the direction of travel
                travel_vec = self.player_pos - old_pos
                if travel_vec.length() > 0:
                    self.player_angle = travel_vec.angle_to(pygame.Vector2(1, 0))

                self._create_travel_effect(old_pos, self.player_pos)
                
                new_dist_to_exit = self.player_pos.distance_to(self.exit_rect.center)
                if new_dist_to_exit < prev_dist_to_exit:
                    reward += 10.0
                else:
                    reward -= 1.0
        
        # Continuous reward for proximity
        new_dist_to_exit = self.player_pos.distance_to(self.exit_rect.center)
        if new_dist_to_exit < prev_dist_to_exit:
            reward += 0.1

        self.score += reward
        self.steps += 1
        self.time_remaining -= 1

        terminated = self._check_termination()
        
        # Apply terminal rewards
        if terminated:
            if self.exit_rect.collidepoint(self.player_pos): # Win
                reward = 100.0
                self.score += reward
            elif self.time_remaining <= 0: # Time out
                reward = -50.0
                self.score += reward
            else: # Collision
                reward = -100.0
                self.score += reward

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _find_target_wormhole(self, direction):
        current_pos = self.wormholes[self.current_wormhole_idx]['pos']
        best_target = None
        min_score = float('inf')

        for i, wh in enumerate(self.wormholes):
            if i == self.current_wormhole_idx:
                continue

            vec = wh['pos'] - current_pos
            if vec.length_squared() == 0:
                continue

            score = 0
            is_valid = False
            
            # Penalize deviation from the desired axis
            deviation_penalty = 2.0 
            
            if direction == 1: # Up
                if vec.y < 0:
                    score = -vec.y + abs(vec.x) * deviation_penalty
                    is_valid = True
            elif direction == 2: # Down
                if vec.y > 0:
                    score = vec.y + abs(vec.x) * deviation_penalty
                    is_valid = True
            elif direction == 3: # Left
                if vec.x < 0:
                    score = -vec.x + abs(vec.y) * deviation_penalty
                    is_valid = True
            elif direction == 4: # Right
                if vec.x > 0:
                    score = vec.x + abs(vec.y) * deviation_penalty
                    is_valid = True

            if is_valid and score < min_score:
                min_score = score
                best_target = i
        
        return best_target

    def _check_termination(self):
        # Collision with obstacles
        for obs in self.obstacles:
            if obs['rect'].collidepoint(self.player_pos):
                # # Sound effect placeholder: explosion
                self._create_explosion(self.player_pos, self.COLOR_OBSTACLE)
                self.game_over = True
                return True
        
        # Reached exit
        if self.exit_rect.collidepoint(self.player_pos):
            # # Sound effect placeholder: victory chime
            self._create_explosion(self.player_pos, self.COLOR_EXIT)
            self.game_over = True
            return True

        # Time up
        if self.time_remaining <= 0:
            self.game_over = True
            return True
            
        return False

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _render_game(self):
        self._update_and_draw_particles()
        self._render_exit()
        self._render_obstacles()
        self._render_wormholes()
        if not self.game_over or self.exit_rect.collidepoint(self.player_pos):
             self._render_player()

    def _render_player(self):
        size = 12
        # Create a triangle pointing right
        points = [
            pygame.Vector2(size, 0),
            pygame.Vector2(-size / 2, -size / 2),
            pygame.Vector2(-size / 2, size / 2)
        ]
        # Rotate points
        rotated_points = [p.rotate(-self.player_angle) + self.player_pos for p in points]
        
        pygame.gfxdraw.aapolygon(self.screen, rotated_points, self.COLOR_PLAYER)
        pygame.gfxdraw.filled_polygon(self.screen, rotated_points, self.COLOR_PLAYER)

    def _render_wormholes(self):
        for i, wh in enumerate(self.wormholes):
            pulse = math.sin(self.steps * 0.1 + wh['pulse_offset']) * 2
            radius = int(wh['radius'] + pulse)
            pos = (int(wh['pos'].x), int(wh['pos'].y))
            
            # Glow effect for the current wormhole
            if i == self.current_wormhole_idx:
                glow_radius = int(radius * 1.8)
                glow_color = tuple(min(255, c + 50) for c in wh['color'])
                pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], glow_radius, (*glow_color, 60))
                pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], glow_radius, (*glow_color, 80))

            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], radius, wh['color'])
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], radius, wh['color'])

    def _render_obstacles(self):
        for obs in self.obstacles:
            obs['angle'] += obs['speed']
            
            # Create original surface
            size = obs['rect'].width
            surf = pygame.Surface((size, size), pygame.SRCALPHA)
            pygame.draw.rect(surf, self.COLOR_OBSTACLE, (0, 0, size, size), border_radius=3)
            pygame.draw.rect(surf, (0,0,0), (5, 5, size-10, size-10), border_radius=2)
            
            # Rotate surface
            rotated_surf = pygame.transform.rotate(surf, obs['angle'])
            new_rect = rotated_surf.get_rect(center=obs['rect'].center)
            
            self.screen.blit(rotated_surf, new_rect.topleft)

    def _render_exit(self):
        # Glow effect
        glow_rect = self.exit_rect.inflate(20, 20)
        s = pygame.Surface(glow_rect.size, pygame.SRCALPHA)
        pygame.draw.rect(s, (*self.COLOR_EXIT, 30), s.get_rect(), border_radius=15)
        self.screen.blit(s, glow_rect.topleft)

        pygame.draw.rect(self.screen, self.COLOR_EXIT, self.exit_rect, border_radius=10)

    def _render_ui(self):
        # Timer
        time_text = f"TIME: {max(0, self.time_remaining // self.FPS):02d}"
        time_surf = self.font_small.render(time_text, True, self.COLOR_TEXT)
        self.screen.blit(time_surf, (20, 15))

        # Score
        score_text = f"SCORE: {int(self.score)}"
        score_surf = self.font_small.render(score_text, True, self.COLOR_TEXT)
        score_rect = score_surf.get_rect(topright=(self.SCREEN_WIDTH - 20, 15))
        self.screen.blit(score_surf, score_rect)

        if self.game_over:
            if self.exit_rect.collidepoint(self.player_pos):
                msg = "SUCCESS"
            elif self.time_remaining <= 0:
                msg = "TIME UP"
            else:
                msg = "GAME OVER"
            
            msg_surf = self.font_large.render(msg, True, self.COLOR_TEXT)
            msg_rect = msg_surf.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(msg_surf, msg_rect)

    def _update_and_draw_particles(self):
        self.particles = [p for p in self.particles if p['lifespan'] > 0]
        for p in self.particles:
            p['pos'] += p['vel']
            p['lifespan'] -= 1
            p['radius'] -= 0.1
            if p['radius'] > 0:
                color = (*p['color'], int(255 * (p['lifespan'] / p['max_lifespan'])))
                pygame.gfxdraw.filled_circle(
                    self.screen, int(p['pos'].x), int(p['pos'].y), int(p['radius']), color
                )

    def _create_explosion(self, position, color):
        for _ in range(50):
            self.particles.append({
                'pos': position.copy(),
                'vel': pygame.Vector2(self.rng.uniform(-3, 3), self.rng.uniform(-3, 3)),
                'radius': self.rng.integers(5, 10),
                'color': color,
                'lifespan': self.rng.integers(20, 40),
                'max_lifespan': 40
            })

    def _create_travel_effect(self, start_pos, end_pos):
        # # Sound effect placeholder: whoosh
        travel_vec = end_pos - start_pos
        for i in range(20):
            frac = i / 19.0
            pos = start_pos + travel_vec * frac
            self.particles.append({
                'pos': pos,
                'vel': pygame.Vector2(0, 0),
                'radius': self.rng.integers(2, 4),
                'color': self.COLOR_PLAYER,
                'lifespan': self.rng.integers(10, 20),
                'max_lifespan': 20
            })

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "time_remaining": self.time_remaining // self.FPS
        }

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
    # Set the video driver to a dummy one for headless execution
    import os
    os.environ["SDL_VIDEODRIVER"] = "dummy"
    
    env = GameEnv()
    obs, info = env.reset()
    print("Initial state:", info)
    
    terminated = False
    total_reward = 0
    for _ in range(1000):
        if terminated:
            break
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
    
    print(f"Random play finished. Final info: {info}, Total Reward: {total_reward}")
    env.close()

    # To visualize the game, you would need a different setup
    # For example, using pygame to display the frames.
    # The following code is for demonstration and requires a display.
    
    # Re-enable video driver for visualization
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]
    
    print("\nStarting interactive visualization...")
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    display_screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Wormhole Navigator")
    
    running = True
    terminated = False
    
    while running:
        action = [0, 0, 0] # Default no-op
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                terminated = False

        if not terminated:
            keys = pygame.key.get_pressed()
            if keys[pygame.K_UP]:
                action[0] = 1
            elif keys[pygame.K_DOWN]:
                action[0] = 2
            elif keys[pygame.K_LEFT]:
                action[0] = 3
            elif keys[pygame.K_RIGHT]:
                action[0] = 4

            obs, reward, terminated, truncated, info = env.step(action)

        # Convert observation back to a Pygame surface for display
        # The observation is (H, W, C), Pygame needs (W, H) surface
        # surfarray.make_surface expects (W, H, C) array
        display_obs = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(display_obs)
        display_screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        env.clock.tick(GameEnv.FPS)

    env.close()