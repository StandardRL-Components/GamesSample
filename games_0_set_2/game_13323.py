import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T20:41:58.250451
# Source Brief: brief_03323.md
# Brief Index: 3323
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    A Gymnasium environment for a marble maze game.

    The agent controls a marble with tilt-like actions, navigating a pseudo-3D
    track to collect gems against a timer. The goal is to collect 15 gems
    (150 points) within 60 seconds.

    **Visuals:**
    The game uses a fixed third-person camera that follows the marble. A
    painter's algorithm (sorting by y-coordinate) and scaling based on a
    simulated z-axis create a sense of depth. Visual effects like shadows,
    a marble glow, and particle bursts on gem collection enhance the experience.

    **Physics:**
    The marble has momentum, acceleration from player input, and friction.
    Falling off the track is a terminal condition.

    **Action Space:** MultiDiscrete([5, 2, 2])
    - `action[0]`: Movement (0: none, 1: up, 2: down, 3: left, 4: right)
    - `action[1]`: Unused (space bar)
    - `action[2]`: Unused (shift key)
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = "Navigate a marble through a pseudo-3D maze, collecting all the gems before time runs out."
    user_guide = "Controls: Use the arrow keys (↑↓←→) to roll the marble and navigate the maze."
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    FPS = 30  # Assumed for physics and timer calculations
    MAX_STEPS = 60 * FPS  # 60 seconds

    # Colors
    COLOR_BG = (20, 30, 50)
    COLOR_TRACK = (100, 150, 250)
    COLOR_TRACK_BORDER = (80, 120, 200)
    COLOR_MARBLE = (255, 80, 80)
    COLOR_MARBLE_GLOW = (255, 150, 150, 32)
    COLOR_GEM = (255, 220, 50)
    COLOR_GEM_GLOW = (255, 220, 50, 48)
    COLOR_SHADOW = (0, 0, 0, 64)
    COLOR_UI_TEXT = (255, 255, 255)
    COLOR_UI_BG = (0, 0, 0, 128)

    # Physics & Gameplay
    MARBLE_ACCELERATION = 0.25
    MARBLE_FRICTION = 0.985
    MARBLE_MAX_SPEED = 5.0
    MARBLE_RADIUS = 12
    GEM_RADIUS = 8
    WIN_SCORE = 150

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

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
        self.font_ui = pygame.font.SysFont("Arial", 20, bold=True)
        self.font_gems = pygame.font.SysFont("Arial", 24, bold=True)

        # --- Game State ---
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.gems_collected = 0

        self.marble_pos = pygame.Vector2(0, 0)
        self.marble_vel = pygame.Vector2(0, 0)
        self.marble_z = 0.0 # For falling effect
        self.is_falling = False
        self.fall_timer = 0

        self.gems = []
        self.particles = []

        self._create_track_and_gems()
        # self.reset() is called by the wrapper/runner
        
        # This check is disabled by default as it runs a step,
        # but can be uncommented for validation.
        # self.validate_implementation()

    def _create_track_and_gems(self):
        """Defines the static layout of the track and initial gem positions."""
        self.track_rects = [
            pygame.Rect(100, 50, 440, 80),
            pygame.Rect(460, 130, 80, 200),
            pygame.Rect(100, 250, 360, 80),
            pygame.Rect(100, 330, 80, 200),
            pygame.Rect(180, 450, 400, 80),
            pygame.Rect(500, 530, 80, 200),
            pygame.Rect(100, 650, 400, 80)
        ]
        self.start_pos = pygame.Vector2(self.track_rects[0].centerx, self.track_rects[0].centery)

        self.initial_gem_positions = [
            (200, 90), (320, 90), (440, 90),
            (500, 180), (500, 280),
            (400, 290), (280, 290), (160, 290),
            (140, 380), (140, 480),
            (250, 490), (370, 490), (490, 490),
            (540, 580), (540, 680)
        ]

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.gems_collected = 0
        self.game_over = False

        self.marble_pos.update(self.start_pos)
        self.marble_vel.update(0, 0)
        self.marble_z = 0.0
        self.is_falling = False
        self.fall_timer = 0

        self.gems = [pygame.Vector2(pos) for pos in self.initial_gem_positions]
        self.particles = []

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # --- 1. Unpack action and update physics ---
        movement = action[0]
        self._update_physics(movement)

        # --- 2. Calculate pre-move state for reward ---
        dist_before_move = self._get_dist_to_nearest_gem()

        # --- 3. Update marble position ---
        self.marble_pos += self.marble_vel

        # --- 4. Game Logic and Reward Calculation ---
        reward = 0
        terminated = False

        # Proximity reward
        dist_after_move = self._get_dist_to_nearest_gem()
        if dist_after_move is not None and dist_before_move is not None:
            if dist_after_move < dist_before_move:
                reward += 0.1 # Moved closer to a gem

        # Gem collection
        collected_gem = self._check_gem_collection()
        if collected_gem:
            reward += 10.0
            self.score += 10
            self.gems_collected += 1
            # sfx: gem_collect.wav

        # Falling off track
        if not self.is_falling and not self._is_on_track():
            self.is_falling = True
            self.fall_timer = self.FPS # 1 second fall animation
            reward -= 20.0
            # sfx: fall.wav
        
        if self.is_falling:
            self.fall_timer -= 1
            self.marble_z -= 0.5 # Visually fall down
            if self.fall_timer <= 0:
                terminated = True
                self.game_over = True

        # --- 5. Update step counter and check termination conditions ---
        self.steps += 1
        
        if self.score >= self.WIN_SCORE:
            reward += 50.0
            terminated = True
            self.game_over = True
            # sfx: win_level.wav
        elif self.steps >= self.MAX_STEPS:
            terminated = True
            self.game_over = True
            # sfx: time_up.wav

        # --- 6. Update particles ---
        self._update_particles()

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _update_physics(self, movement):
        """Applies forces to the marble based on action."""
        acc = pygame.Vector2(0, 0)
        if not self.is_falling:
            if movement == 1: acc.y -= self.MARBLE_ACCELERATION  # Up
            elif movement == 2: acc.y += self.MARBLE_ACCELERATION  # Down
            elif movement == 3: acc.x -= self.MARBLE_ACCELERATION  # Left
            elif movement == 4: acc.x += self.MARBLE_ACCELERATION  # Right

        self.marble_vel += acc
        self.marble_vel *= self.MARBLE_FRICTION

        if self.marble_vel.length() > self.MARBLE_MAX_SPEED:
            self.marble_vel.scale_to_length(self.MARBLE_MAX_SPEED)

    def _get_dist_to_nearest_gem(self):
        if not self.gems:
            return None
        return min(self.marble_pos.distance_to(gem) for gem in self.gems)

    def _check_gem_collection(self):
        """Checks for and handles gem collection."""
        for gem in self.gems[:]:
            if self.marble_pos.distance_to(gem) < self.MARBLE_RADIUS + self.GEM_RADIUS:
                self.gems.remove(gem)
                self._create_particles(gem, self.COLOR_GEM, 20)
                return True
        return False

    def _is_on_track(self):
        """Checks if the marble's center is within any track rectangle."""
        return any(rect.collidepoint(self.marble_pos) for rect in self.track_rects)

    def _create_particles(self, pos, color, count):
        for _ in range(count):
            self.particles.append({
                'pos': pos.copy(),
                'vel': pygame.Vector2(random.uniform(-2, 2), random.uniform(-2, 2)),
                'life': random.randint(10, 20),
                'color': color,
                'radius': random.uniform(1, 4)
            })

    def _update_particles(self):
        for p in self.particles[:]:
            p['pos'] += p['vel']
            p['life'] -= 1
            if p['life'] <= 0:
                self.particles.remove(p)

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        """Renders the main game world with a pseudo-3D perspective."""
        # Camera follows the player
        camera_offset = pygame.Vector2(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2) - self.marble_pos

        # Collect all renderable objects
        render_list = []
        for rect in self.track_rects:
            render_list.append({'type': 'track', 'rect': rect, 'y': rect.centery})
        for gem in self.gems:
            render_list.append({'type': 'gem', 'pos': gem, 'y': gem.y})
        
        # Add marble to the list
        marble_render_y = self.marble_pos.y - self.marble_z * 0.5
        render_list.append({'type': 'marble', 'pos': self.marble_pos, 'y': marble_render_y})

        # Sort by y-coordinate for painter's algorithm (back to front)
        render_list.sort(key=lambda item: item['y'])

        # Draw objects
        for item in render_list:
            if item['type'] == 'track':
                screen_rect = item['rect'].move(camera_offset)
                pygame.draw.rect(self.screen, self.COLOR_TRACK, screen_rect)
                pygame.draw.rect(self.screen, self.COLOR_TRACK_BORDER, screen_rect, 3)
            else:
                pos = item['pos']
                # Apply pseudo-3D projection
                z = self.marble_z if item['type'] == 'marble' else 0
                screen_pos = pos + camera_offset
                screen_y_proj = screen_pos.y - z * 0.5
                scale = max(0.1, 1.0 - z * 0.05)

                if item['type'] == 'gem':
                    self._draw_circle_with_shadow(screen_pos, self.GEM_RADIUS, self.COLOR_GEM, self.COLOR_GEM_GLOW)
                elif item['type'] == 'marble':
                    self._draw_circle_with_shadow(
                        pygame.Vector2(screen_pos.x, screen_y_proj),
                        int(self.MARBLE_RADIUS * scale),
                        self.COLOR_MARBLE,
                        self.COLOR_MARBLE_GLOW
                    )
        
        # Draw particles on top of everything
        for p in self.particles:
            screen_pos = p['pos'] + camera_offset
            pygame.draw.circle(self.screen, p['color'], (int(screen_pos.x), int(screen_pos.y)), int(p['radius']))

    def _draw_circle_with_shadow(self, pos, radius, color, glow_color):
        # Shadow
        shadow_pos = (int(pos.x), int(pos.y + radius * 1.2))
        shadow_rect = pygame.Rect(0, 0, int(radius * 1.8), int(radius * 0.8))
        shadow_rect.center = shadow_pos
        shadow_surface = pygame.Surface(shadow_rect.size, pygame.SRCALPHA)
        pygame.draw.ellipse(shadow_surface, self.COLOR_SHADOW, (0, 0, *shadow_rect.size))
        self.screen.blit(shadow_surface, shadow_rect.topleft)

        # Glow
        pygame.gfxdraw.filled_circle(self.screen, int(pos.x), int(pos.y), int(radius * 1.5), glow_color)
        
        # Main circle
        pygame.gfxdraw.filled_circle(self.screen, int(pos.x), int(pos.y), radius, color)
        pygame.gfxdraw.aacircle(self.screen, int(pos.x), int(pos.y), radius, tuple(c // 2 for c in color))

    def _render_ui(self):
        """Renders the score, timer, and other UI elements."""
        # Score
        score_text = self.font_ui.render(f"SCORE: {self.score}", True, self.COLOR_UI_TEXT)
        score_rect = score_text.get_rect(topleft=(10, 10))
        self.screen.blit(score_text, score_rect)

        # Timer
        time_left = max(0, (self.MAX_STEPS - self.steps) / self.FPS)
        time_text = self.font_ui.render(f"TIME: {time_left:.1f}", True, self.COLOR_UI_TEXT)
        time_rect = time_text.get_rect(topright=(self.SCREEN_WIDTH - 10, 10))
        self.screen.blit(time_text, time_rect)

        # Gem count
        gem_count_text = self.font_gems.render(f"GEMS: {self.gems_collected} / 15", True, self.COLOR_UI_TEXT)
        gem_count_bg_rect = pygame.Rect(0, 0, gem_count_text.get_width() + 20, gem_count_text.get_height() + 10)
        gem_count_bg_rect.centerx = self.SCREEN_WIDTH / 2
        gem_count_bg_rect.bottom = self.SCREEN_HEIGHT - 10
        
        bg_surf = pygame.Surface(gem_count_bg_rect.size, pygame.SRCALPHA)
        bg_surf.fill(self.COLOR_UI_BG)
        self.screen.blit(bg_surf, gem_count_bg_rect)
        
        gem_count_rect = gem_count_text.get_rect(center=gem_count_bg_rect.center)
        self.screen.blit(gem_count_text, gem_count_rect)


    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "gems_collected": self.gems_collected,
            "time_left_seconds": (self.MAX_STEPS - self.steps) / self.FPS
        }

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        """Call this at the end of __init__ to verify implementation."""
        print("Running implementation validation...")
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
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")
        self.reset() # Reset state after validation step


if __name__ == '__main__':
    # --- Human Playable Demo ---
    # To run this, you need to unset the dummy video driver
    # and install pygame with display support.
    # For example:
    # pip install pygame
    # unset SDL_VIDEODRIVER
    # python your_script_name.py
    
    # Check if we can run in display mode
    try:
        del os.environ['SDL_VIDEODRIVER']
        pygame.display.init()
        pygame.font.init()
        display_mode = True
    except (pygame.error, KeyError):
        print("Pygame display not available. Running in headless mode.")
        display_mode = False

    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    if display_mode:
        screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
        pygame.display.set_caption("Marble Maze Gym Environment")
    
    clock = pygame.time.Clock()
    
    terminated = False
    total_reward = 0.0

    print("\n--- Human Controls ---")
    print("Arrow Keys: Move marble")
    print("R: Reset environment")
    print("Q: Quit")
    
    running = True
    while running:
        action = [0, 0, 0] # Default no-op
        
        if display_mode:
            # --- Event Handling ---
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q:
                        running = False
                    if event.key == pygame.K_r:
                        obs, info = env.reset()
                        total_reward = 0.0
                        terminated = False
                        print("--- Environment Reset ---")

            # --- Action Mapping for Human Play ---
            if not terminated:
                keys = pygame.key.get_pressed()
                movement = 0 # No-op
                if keys[pygame.K_UP]: movement = 1
                elif keys[pygame.K_DOWN]: movement = 2
                elif keys[pygame.K_LEFT]: movement = 3
                elif keys[pygame.K_RIGHT]: movement = 4
                
                space_held = 1 if keys[pygame.K_SPACE] else 0
                shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
                
                action = [movement, space_held, shift_held]

        # --- Environment Step ---
        if not terminated:
            obs, reward, term, trunc, info = env.step(action)
            total_reward += reward
            terminated = term or trunc
            
            if terminated:
                print(f"Episode Finished. Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
                if display_mode:
                    print("Press 'R' to play again or 'Q' to quit.")
                else:
                    running = False # End if headless

        # --- Rendering ---
        if display_mode:
            surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
            screen.blit(surf, (0, 0))
            pygame.display.flip()
        
        clock.tick(env.FPS)

    env.close()