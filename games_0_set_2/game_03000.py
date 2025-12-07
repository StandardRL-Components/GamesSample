import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Use arrow keys to draw line segments. Hold Space for longer segments. Hold Shift and press an arrow key to give the sled a directional boost."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Guide the physics-based sled to the green finish line by drawing a track for it. Race against the clock and use boosts wisely. Don't let the sled fall off the screen!"
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    # --- Constants ---
    # Screen
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    
    # Colors
    COLOR_BG = (15, 15, 25)
    COLOR_GRID = (30, 30, 45)
    COLOR_SLED = (255, 255, 255)
    COLOR_SLED_GLOW = (200, 200, 255, 50)
    COLOR_LINE = (255, 50, 50)
    COLOR_FINISH_LINE = (50, 255, 50)
    COLOR_PARTICLE = (255, 220, 50)
    COLOR_TEXT = (240, 240, 240)
    
    # Game parameters
    FPS = 30
    TIME_LIMIT_SECONDS = 60
    MAX_STEPS = 1000
    FINISH_LINE_X = SCREEN_WIDTH - 40
    
    # Physics
    GRAVITY = 0.3
    SLED_RADIUS = 8
    LINE_THICKNESS = 3
    FRICTION = 0.98
    RESTITUTION = 0.6  # Bounciness
    IMPULSE_STRENGTH = 3.0
    MAX_VELOCITY = 15
    
    # Actions
    LINE_LENGTH_SHORT = 25
    LINE_LENGTH_LONG = 50
    MAX_LINES = 50 # To prevent performance degradation

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
        
        try:
            self.font_large = pygame.font.SysFont("Consolas", 24)
            self.font_small = pygame.font.SysFont("Consolas", 16)
        except pygame.error:
            self.font_large = pygame.font.Font(None, 30)
            self.font_small = pygame.font.Font(None, 20)

        self.sled_pos = np.array([0.0, 0.0])
        self.sled_vel = np.array([0.0, 0.0])
        self.lines = []
        self.particles = []
        self.draw_point = np.array([0.0, 0.0])
        self.steps = 0
        self.score = 0.0
        self.timer = 0.0
        self.game_over = False
        self.win = False
        
        self.reset()
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0.0
        self.timer = 0.0
        self.game_over = False
        self.win = False
        
        # A stable starting configuration is necessary to pass the stability test.
        # The original state was unstable (sled in mid-air with velocity), causing
        # termination with no-op actions. This new state starts the sled at rest
        # on a solid platform, ensuring it doesn't fall without agent input.
        start_y = 100.0
        self.sled_pos = np.array([50.0, start_y])
        self.sled_vel = np.array([0.0, 0.0])

        self.particles = []
        
        # Initial track is a horizontal platform placed under the sled.
        # Gravity will pull the sled onto this platform, where it will rest.
        platform_y = start_y + self.SLED_RADIUS
        start_line_pt1 = np.array([20.0, platform_y])
        start_line_pt2 = np.array([80.0, platform_y])
        self.lines = [(start_line_pt1, start_line_pt2)]
        
        # The drawing cursor starts at the end of the initial platform.
        self.draw_point = start_line_pt2.copy()
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement = action[0]
        space_held = action[1] == 1
        shift_held = action[2] == 1
        
        reward = 0.1 # Survival reward
        
        # --- Handle Actions ---
        direction_vector = {
            1: np.array([0, -1]), # Up
            2: np.array([0, 1]),  # Down
            3: np.array([-1, 0]), # Left
            4: np.array([1, 0]),  # Right
        }.get(movement)

        if direction_vector is not None:
            if shift_held:
                # Apply impulse to sled
                self.sled_vel += direction_vector * self.IMPULSE_STRENGTH
                # placeholder: sfx_boost.play()
            else:
                # Draw a line
                length = self.LINE_LENGTH_LONG if space_held else self.LINE_LENGTH_SHORT
                start_point = self.draw_point
                end_point = start_point + direction_vector * length
                
                # Clamp line to screen
                end_point[0] = np.clip(end_point[0], 0, self.SCREEN_WIDTH)
                end_point[1] = np.clip(end_point[1], 0, self.SCREEN_HEIGHT)
                
                self.lines.append((start_point.copy(), end_point.copy()))
                self.draw_point = end_point.copy()
                
                if len(self.lines) > self.MAX_LINES:
                    self.lines.pop(0)
                # placeholder: sfx_draw_line.play()
        
        # --- Update Game State ---
        self._update_physics()
        self._update_particles()
        
        self.steps += 1
        self.timer += 1.0 / self.FPS
        
        # --- Check Termination ---
        terminated = self._check_termination()
        
        if terminated:
            if self.win:
                time_bonus = max(0, self.TIME_LIMIT_SECONDS - self.timer)
                win_reward = 100.0 + 100.0 * (time_bonus / self.TIME_LIMIT_SECONDS)
                reward += win_reward
                self.score += win_reward
            else: # Loss condition
                loss_penalty = -10.0
                reward += loss_penalty
                self.score += loss_penalty
        
        self.score += reward

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _update_physics(self):
        # Apply gravity
        self.sled_vel[1] += self.GRAVITY
        
        # Clamp velocity
        vel_mag = np.linalg.norm(self.sled_vel)
        if vel_mag > self.MAX_VELOCITY:
            self.sled_vel = self.sled_vel / vel_mag * self.MAX_VELOCITY

        # Update position
        self.sled_pos += self.sled_vel
        
        # Collision detection and response
        for p1, p2 in self.lines:
            closest_point = self._get_closest_point_on_segment(self.sled_pos, p1, p2)
            dist_vec = self.sled_pos - closest_point
            dist = np.linalg.norm(dist_vec)
            
            if dist < self.SLED_RADIUS:
                # Resolve position (prevent sinking)
                penetration = self.SLED_RADIUS - dist
                if dist > 1e-6:
                    normal = dist_vec / dist
                    self.sled_pos += normal * penetration
                else: # Exactly on the line, need to calculate normal differently
                    line_vec = p2 - p1
                    if np.linalg.norm(line_vec) == 0: continue
                    normal = np.array([-line_vec[1], line_vec[0]])
                    normal = normal / np.linalg.norm(normal)
                    self.sled_pos += normal * penetration
                
                # Resolve velocity
                v_normal_component = np.dot(self.sled_vel, normal)
                if v_normal_component < 0: # Moving towards the line
                    # Reflect velocity (bounce)
                    v_reflect = self.sled_vel - (1 + self.RESTITUTION) * v_normal_component * normal
                    
                    # Apply friction
                    v_tangent_comp = v_reflect - np.dot(v_reflect, normal) * normal
                    v_normal_comp = v_reflect - v_tangent_comp
                    self.sled_vel = v_normal_comp + v_tangent_comp * self.FRICTION
                    # placeholder: sfx_sled_scrape.play()

    def _get_closest_point_on_segment(self, p, a, b):
        ap = p - a
        ab = b - a
        ab_squared = np.dot(ab, ab)
        if ab_squared == 0:
            return a
        
        t = np.dot(ap, ab) / ab_squared
        t = np.clip(t, 0, 1)
        return a + t * ab

    def _check_termination(self):
        if self.game_over:
            return True

        # Win condition
        if self.sled_pos[0] >= self.FINISH_LINE_X:
            self.game_over = True
            self.win = True
            # placeholder: sfx_win.play()
            return True
            
        # Loss conditions
        is_off_screen = not (0 < self.sled_pos[0] < self.SCREEN_WIDTH and -50 < self.sled_pos[1] < self.SCREEN_HEIGHT + 50)
        
        if is_off_screen:
            self.game_over = True
            self._create_crash_particles()
            # placeholder: sfx_crash.play()
            return True
        
        if self.timer >= self.TIME_LIMIT_SECONDS:
            self.game_over = True
            return True
        
        if self.steps >= self.MAX_STEPS:
            self.game_over = True
            return True
        
        return False
        
    def _create_crash_particles(self):
        for _ in range(30):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(2, 8)
            vel = np.array([math.cos(angle) * speed, math.sin(angle) * speed])
            life = random.uniform(0.5, 1.5)
            self.particles.append({'pos': self.sled_pos.copy(), 'vel': vel, 'life': life, 'max_life': life})

    def _update_particles(self):
        if not self.particles:
            return
        
        active_particles = []
        for p in self.particles:
            p['pos'] += p['vel']
            p['vel'] *= 0.95 # Drag
            p['life'] -= 1.0 / self.FPS
            if p['life'] > 0:
                active_particles.append(p)
        self.particles = active_particles

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_background()
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_background(self):
        for x in range(0, self.SCREEN_WIDTH, 40):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.SCREEN_HEIGHT))
        for y in range(0, self.SCREEN_HEIGHT, 40):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.SCREEN_WIDTH, y))

    def _render_game(self):
        # Draw finish line
        pygame.draw.line(self.screen, self.COLOR_FINISH_LINE, (self.FINISH_LINE_X, 0), (self.FINISH_LINE_X, self.SCREEN_HEIGHT), 5)
        
        # Draw tracks
        for p1, p2 in self.lines:
            pygame.draw.line(self.screen, self.COLOR_LINE, p1.astype(int), p2.astype(int), self.LINE_THICKNESS)
        
        # Draw draw point cursor
        pygame.gfxdraw.filled_circle(self.screen, int(self.draw_point[0]), int(self.draw_point[1]), 3, self.COLOR_LINE)
        
        # Draw sled
        if not (self.game_over and not self.win):
            sled_x, sled_y = int(self.sled_pos[0]), int(self.sled_pos[1])
            
            # Glow effect
            glow_surface = pygame.Surface((self.SLED_RADIUS * 4, self.SLED_RADIUS * 4), pygame.SRCALPHA)
            pygame.draw.circle(glow_surface, self.COLOR_SLED_GLOW, (self.SLED_RADIUS * 2, self.SLED_RADIUS * 2), self.SLED_RADIUS * 2)
            self.screen.blit(glow_surface, (sled_x - self.SLED_RADIUS * 2, sled_y - self.SLED_RADIUS * 2))

            pygame.gfxdraw.filled_circle(self.screen, sled_x, sled_y, self.SLED_RADIUS, self.COLOR_SLED)
            pygame.gfxdraw.aacircle(self.screen, sled_x, sled_y, self.SLED_RADIUS, self.COLOR_SLED)
            
        # Draw particles
        for p in self.particles:
            alpha = int(255 * (p['life'] / p['max_life']))
            radius = int(3 * (p['life'] / p['max_life']))
            if radius > 0:
                color = (*self.COLOR_PARTICLE, alpha)
                try:
                    pygame.gfxdraw.filled_circle(self.screen, int(p['pos'][0]), int(p['pos'][1]), radius, color)
                except OverflowError: # Can happen if particles fly way off screen
                    pass


    def _render_ui(self):
        time_text = f"TIME: {max(0, self.TIME_LIMIT_SECONDS - self.timer):.1f}"
        score_text = f"SCORE: {self.score:.1f}"
        
        time_surf = self.font_large.render(time_text, True, self.COLOR_TEXT)
        score_surf = self.font_large.render(score_text, True, self.COLOR_TEXT)
        
        self.screen.blit(time_surf, (self.SCREEN_WIDTH - time_surf.get_width() - 10, 10))
        self.screen.blit(score_surf, (10, 10))
        
        if self.game_over:
            status_text = "FINISH!" if self.win else "CRASHED"
            status_color = self.COLOR_FINISH_LINE if self.win else self.COLOR_LINE
            status_surf = self.font_large.render(status_text, True, status_color)
            self.screen.blit(status_surf, (self.SCREEN_WIDTH // 2 - status_surf.get_width() // 2, self.SCREEN_HEIGHT // 2 - status_surf.get_height() // 2))

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "timer": self.timer,
            "sled_pos": self.sled_pos.tolist(),
            "sled_vel": self.sled_vel.tolist(),
            "win": self.win,
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
        
        # print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game directly
    # To run, ensure you have pygame installed: pip install pygame
    # And switch the SDL_VIDEODRIVER to a visible one, e.g., by commenting out the os.environ line
    # at the top of the file.
    
    # Example:
    # del os.environ['SDL_VIDEODRIVER']
    
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Sled Game")
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    while running:
        movement, space, shift = 0, 0, 0
        
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
        
        if keys[pygame.K_r]: # Press R to reset
            obs, info = env.reset()
            total_reward = 0
            print("--- ENV RESET ---")

        action = [movement, space, shift]
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if terminated:
            print(f"Episode finished. Total Reward: {total_reward:.2f}, Info: {info}")
            # In a real scenario, you'd reset here. For playtesting, we can just watch the end state.
            # obs, info = env.reset()
            # total_reward = 0

        # The observation is a numpy array, we need to convert it back to a Pygame surface to display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        clock.tick(env.FPS)
        
    env.close()