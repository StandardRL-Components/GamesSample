import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T16:56:28.668520
# Source Brief: brief_01310.md
# Brief Index: 1310
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    Navigate a neon cityscape by rhythmically activating magnetically-placed portals,
    dodging reality-warping glitches to unlock new portals and rhythms.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Navigate a neon cityscape by rhythmically activating magnetically-placed portals, "
        "dodging reality-warping glitches to unlock new portals and rhythms."
    )
    user_guide = (
        "Use Shift to switch polarity and Space to activate portals at the peak of the rhythm. "
        "Arrow keys are not used."
    )
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    MAX_STEPS = 2000
    TIME_LIMIT_SECONDS = 60 # MAX_STEPS / 30 FPS
    
    # Colors
    COLOR_BG = (10, 5, 25)
    COLOR_GRID = (30, 20, 50)
    COLOR_PLAYER = (255, 255, 0)
    COLOR_PLAYER_GLOW = (255, 255, 0, 50)
    COLOR_POSITIVE = (0, 150, 255)
    COLOR_NEGATIVE = (255, 50, 50)
    COLOR_EXIT = (50, 255, 150)
    COLOR_GLITCH = (180, 0, 255)
    COLOR_TEXT = (220, 220, 240)
    
    # Physics & Gameplay
    PLAYER_RADIUS = 10
    PORTAL_RADIUS = 15
    MAGNETIC_FORCE = 2.0
    PLAYER_DRAG = 0.95
    RHYTHM_BASE_SPEED = 0.05
    PORTAL_ACTIVATION_WINDOW = 0.9  # Must be > this value (peak of pulse)

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Gymnasium Spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame Setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.Font(None, 28)
        self.font_polarity = pygame.font.Font(None, 20)

        # Game Progression State (persists across resets)
        self.successful_exits = 0
        
        # Initialize state variables
        self.player_pos = None
        self.player_vel = None
        self.player_polarity = None # True: Positive (Blue), False: Negative (Red)
        self.portals = None
        self.glitches = None
        self.particles = None
        self.steps = None
        self.score = None
        self.game_over = None
        self.timer = None
        self.rhythm_timer = None
        self.rhythm_speed = None
        self.glitch_spawn_chance = None
        self.prev_space_held = None
        self.prev_shift_held = None
        self.dist_to_exit_prev = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Player state
        self.player_pos = pygame.Vector2(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2)
        self.player_vel = pygame.Vector2(0, 0)
        self.player_polarity = self.np_random.choice([True, False])
        
        # Game state
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.timer = self.TIME_LIMIT_SECONDS * 30 # Assuming 30 FPS
        self.rhythm_timer = 0
        self.glitch_spawn_chance = 0.01
        
        # Input state
        self.prev_space_held = False
        self.prev_shift_held = False

        # Progression-based difficulty
        self.unlocked_portals_count = 1 + self.successful_exits // 2
        self.rhythm_speed = self.RHYTHM_BASE_SPEED * (1.05 ** (self.successful_exits // 2))

        # Entity lists
        self.portals = []
        self.glitches = []
        self.particles = []
        
        self._setup_level()
        
        exit_portal = self._get_exit_portal()
        if exit_portal:
            self.dist_to_exit_prev = self.player_pos.distance_to(exit_portal['pos'])
        else:
            self.dist_to_exit_prev = float('inf')

        return self._get_observation(), self._get_info()

    def _setup_level(self):
        # Create Exit Portal
        exit_pos = self._get_random_pos_away_from_center(150)
        self.portals.append({'pos': exit_pos, 'polarity': None, 'is_exit': True})

        # Create Normal Portals
        for _ in range(self.unlocked_portals_count):
            portal_pos = self._get_random_pos_away_from_center(50)
            portal_polarity = self.np_random.choice([True, False])
            self.portals.append({'pos': portal_pos, 'polarity': portal_polarity, 'is_exit': False})

    def _get_random_pos_away_from_center(self, min_dist):
        while True:
            pos = pygame.Vector2(
                self.np_random.uniform(self.PORTAL_RADIUS, self.SCREEN_WIDTH - self.PORTAL_RADIUS),
                self.np_random.uniform(self.PORTAL_RADIUS, self.SCREEN_HEIGHT - self.PORTAL_RADIUS)
            )
            if pos.distance_to(pygame.Vector2(self.SCREEN_WIDTH/2, self.SCREEN_HEIGHT/2)) > min_dist:
                return pos

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # --- Action Handling ---
        movement = action[0]  # Unused, per brief
        space_held = action[1] == 1
        shift_held = action[2] == 1
        
        reward = -0.01 # Small penalty for each step to encourage speed
        
        # Rising edge detection for discrete actions
        space_pressed = space_held and not self.prev_space_held
        shift_pressed = shift_held and not self.prev_shift_held
        
        if shift_pressed:
            self.player_polarity = not self.player_polarity
            # sfx: polarity_switch.wav
            self._spawn_particles(self.player_pos, 10, self.COLOR_POSITIVE if self.player_polarity else self.COLOR_NEGATIVE)

        # --- Game Logic Update ---
        self._update_physics()
        
        if space_pressed:
            reward += self._handle_portal_activation()

        self._update_glitches()
        self._update_particles()
        
        # --- State Updates ---
        self.steps += 1
        self.timer -= 1
        self.rhythm_timer += self.rhythm_speed
        if self.steps % 100 == 0:
            self.glitch_spawn_chance = min(0.1, self.glitch_spawn_chance + 0.01)

        # --- Reward Calculation ---
        reward += self._calculate_reward()

        # --- Termination Check ---
        terminated = self._check_termination()
        if terminated and not self.game_over: # Handle termination from this step
            if self.timer <= 0:
                reward = -100
                # sfx: time_out.wav
            self.game_over = True
        
        self.score += reward
        self.prev_space_held = space_held
        self.prev_shift_held = shift_held
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _update_physics(self):
        # Magnetic forces from portals
        force_vector = pygame.Vector2(0, 0)
        for portal in self.portals:
            if portal['is_exit']: continue
            
            vec_to_player = self.player_pos - portal['pos']
            dist = vec_to_player.length()
            
            if dist > 1:
                # Same polarity repels, opposite attracts
                force_direction = 1 if portal['polarity'] == self.player_polarity else -1
                
                # Force stronger when closer: 1/dist
                strength = force_direction * self.MAGNETIC_FORCE / dist
                force_vector += vec_to_player.normalize() * strength

        self.player_vel += force_vector
        self.player_vel *= self.PLAYER_DRAG
        self.player_pos += self.player_vel

        # Toroidal world wrap-around
        self.player_pos.x %= self.SCREEN_WIDTH
        self.player_pos.y %= self.SCREEN_HEIGHT

    def _handle_portal_activation(self):
        # Find nearest portal
        nearest_portal = None
        min_dist = float('inf')
        
        for portal in self.portals:
            dist = self.player_pos.distance_to(portal['pos'])
            if dist < min_dist:
                min_dist = dist
                nearest_portal = portal
        
        if nearest_portal:
            pulse = (math.sin(self.rhythm_timer) + 1) / 2
            is_active_time = pulse > self.PORTAL_ACTIVATION_WINDOW
            
            # Can activate if close enough and at the right rhythm
            if min_dist < self.PORTAL_RADIUS + self.PLAYER_RADIUS and is_active_time:
                # sfx: portal_jump.wav
                self._spawn_particles(self.player_pos, 30, self.COLOR_TEXT)
                self.player_pos = pygame.Vector2(nearest_portal['pos'])
                self.player_vel = pygame.Vector2(0, 0)
                
                if nearest_portal['is_exit']:
                    self.game_over = True
                    self.successful_exits += 1
                    # sfx: level_complete.wav
                    return 100 # Goal reward
                else:
                    return 5 # Portal activation reward
        return 0

    def _update_glitches(self):
        # Spawn new glitches
        if self.np_random.random() < self.glitch_spawn_chance:
            pos = self._get_random_pos_away_from_center(0)
            size = self.np_random.uniform(10, 20)
            max_size = size * self.np_random.uniform(3, 6)
            lifetime = self.np_random.integers(100, 200)
            self.glitches.append({
                'rect': pygame.Rect(pos.x - size/2, pos.y - size/2, size, size),
                'max_size': max_size,
                'lifetime': lifetime,
                'initial_lifetime': lifetime
            })
        
        # Update and check existing glitches
        for glitch in self.glitches[:]:
            glitch['lifetime'] -= 1
            if glitch['lifetime'] <= 0:
                self.glitches.remove(glitch)
                continue
            
            # Grow glitch
            growth_factor = 1 - (glitch['lifetime'] / glitch['initial_lifetime'])
            new_size = glitch['rect'].width + (glitch['max_size'] - glitch['rect'].width) * growth_factor * 0.1
            glitch['rect'].inflate_ip(new_size - glitch['rect'].width, new_size - glitch['rect'].height)
            
            # Check for collision
            if glitch['rect'].collidepoint(self.player_pos):
                self.game_over = True
                self.score -= 10 # Event penalty
                # sfx: glitch_hit.wav
                self._spawn_particles(self.player_pos, 50, self.COLOR_GLITCH)
                return

    def _update_particles(self):
        for p in self.particles[:]:
            p['pos'] += p['vel']
            p['life'] -= 1
            if p['life'] <= 0:
                self.particles.remove(p)

    def _spawn_particles(self, pos, count, color):
        for _ in range(count):
            self.particles.append({
                'pos': pygame.Vector2(pos),
                'vel': pygame.Vector2(self.np_random.uniform(-2, 2), self.np_random.uniform(-2, 2)),
                'life': self.np_random.integers(10, 25),
                'color': color
            })

    def _calculate_reward(self):
        exit_portal = self._get_exit_portal()
        if not exit_portal: return 0

        dist_to_exit = self.player_pos.distance_to(exit_portal['pos'])
        
        reward = 0
        if dist_to_exit < self.dist_to_exit_prev:
            reward += 0.1  # Closer to exit
        else:
            reward -= 0.5  # Further from exit

        self.dist_to_exit_prev = dist_to_exit
        return reward

    def _check_termination(self):
        return self.game_over or self.steps >= self.MAX_STEPS or self.timer <= 0

    def _get_exit_portal(self):
        for p in self.portals:
            if p['is_exit']:
                return p
        return None

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_background()
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "timer": self.timer / 30,
            "successful_exits": self.successful_exits
        }

    def _render_background(self):
        # Grid
        for x in range(0, self.SCREEN_WIDTH, 40):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.SCREEN_HEIGHT))
        for y in range(0, self.SCREEN_HEIGHT, 40):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.SCREEN_WIDTH, y))
        
        # Faint building outlines could be added here for more detail

    def _render_game(self):
        # Render Glitches
        for glitch in self.glitches:
            alpha = 100 + 100 * math.sin(pygame.time.get_ticks() * 0.05)
            s = pygame.Surface(glitch['rect'].size, pygame.SRCALPHA)
            color = self.COLOR_GLITCH + (int(alpha),)
            pygame.gfxdraw.box(s, s.get_rect(), color)
            self.screen.blit(s, glitch['rect'].topleft)

        # Render Portals
        pulse = (math.sin(self.rhythm_timer) + 1) / 2 # 0 to 1
        for portal in self.portals:
            pos = (int(portal['pos'].x), int(portal['pos'].y))
            
            if portal['is_exit']:
                color = self.COLOR_EXIT
            else:
                color = self.COLOR_POSITIVE if portal['polarity'] else self.COLOR_NEGATIVE
            
            # Glow effect
            glow_radius = int(self.PORTAL_RADIUS * (1.2 + 0.3 * pulse))
            glow_color = color + (30,)
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], glow_radius, glow_color)
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], glow_radius, glow_color)

            # Core portal
            radius = int(self.PORTAL_RADIUS * (0.8 + 0.2 * pulse))
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], radius, color)
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], radius, color)

            # Activation indicator
            if pulse > self.PORTAL_ACTIVATION_WINDOW:
                pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], radius + 3, self.COLOR_TEXT)
    
        # Render Particles
        for p in self.particles:
            alpha = max(0, min(255, int(255 * (p['life'] / 25.0))))
            color = p['color'] + (alpha,) if len(p['color']) == 3 else p['color']
            pygame.draw.circle(self.screen, color, (int(p['pos'].x), int(p['pos'].y)), 2)

        # Render Player
        player_pos_int = (int(self.player_pos.x), int(self.player_pos.y))
        
        # Glow
        pygame.gfxdraw.filled_circle(self.screen, player_pos_int[0], player_pos_int[1], self.PLAYER_RADIUS + 5, self.COLOR_PLAYER_GLOW)
        
        # Core
        pygame.gfxdraw.filled_circle(self.screen, player_pos_int[0], player_pos_int[1], self.PLAYER_RADIUS, self.COLOR_PLAYER)
        pygame.gfxdraw.aacircle(self.screen, player_pos_int[0], player_pos_int[1], self.PLAYER_RADIUS, self.COLOR_PLAYER)
        
        # Polarity indicator
        polarity_color = self.COLOR_POSITIVE if self.player_polarity else self.COLOR_NEGATIVE
        pygame.draw.circle(self.screen, polarity_color, player_pos_int, self.PLAYER_RADIUS // 2)

    def _render_ui(self):
        # Score
        score_text = self.font_ui.render(f"Score: {int(self.score)}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))

        # Timer
        time_str = f"Time: {max(0, self.timer / 30):.1f}"
        timer_text = self.font_ui.render(time_str, True, self.COLOR_TEXT)
        self.screen.blit(timer_text, (self.SCREEN_WIDTH - timer_text.get_width() - 10, 10))

        # Rhythm Bar
        pulse = (math.sin(self.rhythm_timer) + 1) / 2
        bar_width = 200
        bar_height = 10
        bar_x = (self.SCREEN_WIDTH - bar_width) / 2
        bar_y = self.SCREEN_HEIGHT - 20
        
        pygame.draw.rect(self.screen, self.COLOR_GRID, (bar_x, bar_y, bar_width, bar_height), 1)
        indicator_pos = bar_x + pulse * bar_width
        pygame.draw.rect(self.screen, self.COLOR_TEXT, (indicator_pos - 2, bar_y - 2, 4, bar_height + 4))

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        # This method is for initial development and can be removed or commented out.
        print("Running implementation validation...")
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test reset to get initial observation
        obs, info = self.reset()
        
        # Test observation space  
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert obs.dtype == np.uint8
        
        # Test reset return types
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

if __name__ == '__main__':
    # This block allows you to play the game manually for testing
    # It will not run in the headless environment
    try:
        os.environ["SDL_VIDEODRIVER"] = "x11"
        env = GameEnv()
        obs, info = env.reset()
        
        running = True
        game_window = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
        pygame.display.set_caption("Rhythm Portal Navigator")
        clock = pygame.time.Clock()

        total_reward = 0
        
        while running:
            movement = 0 # No-op
            space = 0
            shift = 0

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
            
            keys = pygame.key.get_pressed()
            if keys[pygame.K_UP]: movement = 1
            elif keys[pygame.K_DOWN]: movement = 2
            elif keys[pygame.K_LEFT]: movement = 3
            elif keys[pygame.K_RIGHT]: movement = 4
            
            if keys[pygame.K_SPACE]: space = 1
            if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift = 1
            
            action = [movement, space, shift]
            
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward

            # Render the observation to the game window
            surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
            game_window.blit(surf, (0, 0))
            pygame.display.flip()
            
            if terminated or truncated:
                print(f"Episode finished. Total Reward: {total_reward:.2f}, Info: {info}")
                total_reward = 0
                obs, info = env.reset()
                
            clock.tick(30) # Run at 30 FPS

        env.close()
    except pygame.error as e:
        print(f"Could not run graphical test: {e}")
        print("This is expected in a headless environment.")
        # Test the headless version
        env = GameEnv()
        env.validate_implementation()
        env.close()