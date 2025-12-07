
# Generated: 2025-08-28T02:03:04.096488
# Source Brief: brief_01581.md
# Brief Index: 1581

        
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

    # Must be a short, user-facing control string:
    user_guide = "Controls: Press space to jump."

    # Must be a user-facing description of the game:
    game_description = (
        "A minimalist platformer where the player uses a single button to jump between "
        "procedurally generated platforms, aiming to reach the top while collecting stars."
    )

    # Frames auto-advance for real-time gameplay.
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    FPS = 30

    # Colors
    COLOR_BG_TOP = (20, 30, 80)
    COLOR_BG_BOTTOM = (60, 80, 160)
    COLOR_PLAYER = (255, 255, 255)
    COLOR_PLATFORM = (150, 150, 170)
    COLOR_GOAL_PLATFORM = (180, 220, 180)
    COLOR_STAR = (255, 223, 0)
    COLOR_TEXT = (240, 240, 240)
    COLOR_TEXT_SHADOW = (20, 20, 20)

    # Physics
    GRAVITY = 0.8
    JUMP_STRENGTH = -14
    PLAYER_SIZE = 16

    # Gameplay
    MAX_STAGES = 3
    TIME_LIMIT_SECONDS = 60
    PLATFORMS_PER_STAGE = 30
    STAR_CHANCE = 0.4

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Gymnasium spaces
        self.observation_space = Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.Font(None, 50)
        self.font_medium = pygame.font.Font(None, 36)
        self.font_small = pygame.font.Font(None, 24)

        # Pre-calculate background gradient
        self.background = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        for y in range(self.SCREEN_HEIGHT):
            interp = y / self.SCREEN_HEIGHT
            color = [
                self.COLOR_BG_TOP[i] * (1 - interp) + self.COLOR_BG_BOTTOM[i] * interp
                for i in range(3)
            ]
            pygame.draw.line(self.background, color, (0, y), (self.SCREEN_WIDTH, y))
        
        # Pre-calculate star points
        self.star_points = []
        for i in range(5):
            angle = math.radians(i * 72 - 90)
            self.star_points.append((math.cos(angle), math.sin(angle)))

        # Initialize state variables
        self.player_pos = None
        self.player_vel = None
        self.on_ground = None
        self.platforms = None
        self.stars = None
        self.particles = None
        self.camera_y = None
        self.steps = None
        self.score = None
        self.stage = None
        self.timer = None
        self.time_limit_frames = self.TIME_LIMIT_SECONDS * self.FPS
        self.stage_platforms_generated = None
        self.message = None
        self.message_timer = None
        self.last_jump_risky = None

        # This will be called in reset() but needed for validate_implementation
        self.np_random = np.random.default_rng()

        self.validate_implementation()

    def _start_new_stage(self):
        self.player_pos = [self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT - 50]
        self.player_vel = [0, 0]
        self.on_ground = True
        self.camera_y = 0
        self.platforms = []
        self.stars = []
        self.stage_platforms_generated = 0
        self.timer = self.time_limit_frames
        
        # Initial platform
        initial_platform = pygame.Rect(self.SCREEN_WIDTH / 2 - 50, self.SCREEN_HEIGHT - 20, 100, 20)
        self.platforms.append(initial_platform)
        self._generate_initial_platforms()
        
        if self.stage > 1:
            self.message = f"Stage {self.stage}"
            self.message_timer = self.FPS * 2 # 2 seconds
        
    def _generate_initial_platforms(self):
        while len(self.platforms) < self.PLATFORMS_PER_STAGE:
            self._generate_next_platform()

    def _generate_next_platform(self):
        last_platform = self.platforms[-1]
        gap_multiplier = 1.0 + 0.05 * (self.stage - 1)
        
        min_y_gap = 80
        max_y_gap = 180
        
        new_y = last_platform.y - self.np_random.uniform(min_y_gap, max_y_gap)
        
        max_offset = 120 * gap_multiplier
        new_x = last_platform.centerx + self.np_random.uniform(-max_offset, max_offset)
        
        new_width = self.np_random.uniform(60, 150)
        new_x -= new_width / 2
        
        # Prevent platforms from going off-screen
        new_x = max(20, min(new_x, self.SCREEN_WIDTH - new_width - 20))
        
        is_goal = self.stage_platforms_generated == self.PLATFORMS_PER_STAGE - 1
        
        new_platform = pygame.Rect(new_x, new_y, new_width if not is_goal else self.SCREEN_WIDTH, 20)
        self.platforms.append(new_platform)
        
        # Add a star?
        if not is_goal and self.np_random.random() < self.STAR_CHANCE:
            star_pos = (new_platform.centerx, new_platform.top - 15)
            self.stars.append({'pos': star_pos, 'rect': pygame.Rect(star_pos[0]-10, star_pos[1]-10, 20, 20)})

        self.stage_platforms_generated += 1

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.stage = 1
        self.message = None
        self.message_timer = 0
        self.particles = []
        self.last_jump_risky = 0
        
        self._start_new_stage()
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        self.clock.tick(self.FPS)
        self.steps += 1
        reward = 0
        
        # Unpack action
        space_pressed = action[1] == 1

        if self.message_timer > 0:
            self.message_timer -= 1
            if self.message_timer <= 0:
                self.message = None
        else:
            # --- Game Logic ---
            self.timer -= 1
            
            # Player input
            if space_pressed and self.on_ground:
                self.player_vel[1] = self.JUMP_STRENGTH
                self.on_ground = False
                # sfx: jump
                self._create_particles(self.player_pos, 10, self.COLOR_PLAYER, 'up')
                self.last_jump_risky = 1 if self.platforms[0].width < 80 else (-0.02 if self.platforms[0].width > 120 else 0)


            # Player physics
            self.player_vel[1] += self.GRAVITY
            self.player_pos[1] += self.player_vel[1]
            
            player_rect = pygame.Rect(self.player_pos[0] - self.PLAYER_SIZE/2, self.player_pos[1] - self.PLAYER_SIZE, self.PLAYER_SIZE, self.PLAYER_SIZE)

            # Collision detection
            self.on_ground = False
            if self.player_vel[1] > 0:
                for i, p in enumerate(self.platforms):
                    if player_rect.colliderect(p) and player_rect.bottom < p.centery:
                        self.player_pos[1] = p.top
                        self.player_vel[1] = 0
                        self.on_ground = True
                        # sfx: land
                        self._create_particles((self.player_pos[0], p.top), 15, self.COLOR_PLATFORM, 'out')
                        
                        # Apply jump risk reward/penalty on landing
                        reward += self.last_jump_risky
                        self.last_jump_risky = 0

                        # Check for stage completion
                        is_goal = (i == len(self.platforms) - 1 and self.stage_platforms_generated >= self.PLATFORMS_PER_STAGE)
                        if is_goal:
                            if self.stage < self.MAX_STAGES:
                                self.stage += 1
                                reward += 100
                                # sfx: stage_clear
                                self._start_new_stage()
                            else: # Game Won
                                reward += 300
                                self.message = "You Win!"
                                self.message_timer = self.FPS * 5
                                return self._get_observation(), reward, True, False, self._get_info()
                        break

            # Star collection
            for star in self.stars[:]:
                if player_rect.colliderect(star['rect']):
                    self.stars.remove(star)
                    self.score += 10
                    reward += 5
                    # sfx: star_collect
                    self._create_particles(star['pos'], 20, self.COLOR_STAR, 'sparkle')

            # Camera control (scrolling)
            scroll_threshold = self.SCREEN_HEIGHT * 0.5
            if self.player_pos[1] < scroll_threshold:
                scroll = scroll_threshold - self.player_pos[1]
                self.player_pos[1] += scroll
                self.camera_y += scroll
                for p in self.platforms:
                    p.y += scroll
                for s in self.stars:
                    s['pos'] = (s['pos'][0], s['pos'][1] + scroll)
                    s['rect'].y += scroll
                for part in self.particles:
                    part['pos'][1] += scroll
            
            # Continuous rewards
            # Reward for upward velocity, penalize for downward
            if self.player_vel[1] < 0:
                reward += 0.1
            else:
                reward -= 0.2

            # Culling old platforms and generating new ones
            self.platforms = [p for p in self.platforms if p.top < self.SCREEN_HEIGHT + 50]
            self.stars = [s for s in self.stars if s['rect'].top < self.SCREEN_HEIGHT + 50]
            if len(self.platforms) > 0 and self.platforms[-1].y > -50 and self.stage_platforms_generated < self.PLATFORMS_PER_STAGE:
                 self._generate_next_platform()
        
        # --- Particles ---
        self._update_particles()
        
        # --- Termination ---
        terminated = False
        if self.player_pos[1] > self.SCREEN_HEIGHT + 50:
            terminated = True
            reward = -50
            self.message = "Game Over"
            self.message_timer = self.FPS * 3
            # sfx: fall
        elif self.timer <= 0:
            terminated = True
            reward = -50
            self.message = "Time's Up!"
            self.message_timer = self.FPS * 3
            # sfx: time_up

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _create_particles(self, pos, count, color, p_type):
        for _ in range(count):
            if p_type == 'up':
                vel = [self.np_random.uniform(-1, 1), self.np_random.uniform(-4, -1)]
            elif p_type == 'out':
                angle = self.np_random.uniform(0, math.pi) # upper semi-circle
                speed = self.np_random.uniform(1, 4)
                vel = [math.cos(angle) * speed, -math.sin(angle) * speed]
            elif p_type == 'sparkle':
                angle = self.np_random.uniform(0, 2 * math.pi)
                speed = self.np_random.uniform(1, 5)
                vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            
            self.particles.append({
                'pos': list(pos),
                'vel': vel,
                'life': self.np_random.integers(15, 30),
                'color': color,
                'radius': self.np_random.uniform(2, 5)
            })

    def _update_particles(self):
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['vel'][1] += 0.1 # particle gravity
            p['life'] -= 1
            p['radius'] -= 0.1
        self.particles = [p for p in self.particles if p['life'] > 0 and p['radius'] > 0]
    
    def _get_observation(self):
        self.screen.blit(self.background, (0, 0))
        
        # Render platforms
        for i, p in enumerate(self.platforms):
            is_goal = (i == len(self.platforms) - 1 and self.stage_platforms_generated >= self.PLATFORMS_PER_STAGE)
            color = self.COLOR_GOAL_PLATFORM if is_goal else self.COLOR_PLATFORM
            pygame.draw.rect(self.screen, color, p, border_radius=3)

        # Render stars
        for s in self.stars:
            points = [(s['pos'][0] + pt[0] * 10, s['pos'][1] + pt[1] * 10) for pt in self.star_points]
            pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_STAR)
            pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_STAR)
            
        # Render particles
        for p in self.particles:
            alpha = max(0, min(255, int(255 * (p['life'] / 20))))
            color = (*p['color'], alpha)
            temp_surf = pygame.Surface((p['radius']*2, p['radius']*2), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, color, (p['radius'], p['radius']), p['radius'])
            self.screen.blit(temp_surf, (int(p['pos'][0] - p['radius']), int(p['pos'][1] - p['radius'])))

        # Render player
        player_rect = pygame.Rect(0, 0, self.PLAYER_SIZE, self.PLAYER_SIZE)
        player_rect.center = self.player_pos
        # Glow effect
        glow_surf = pygame.Surface((self.PLAYER_SIZE*2, self.PLAYER_SIZE*2), pygame.SRCALPHA)
        pygame.draw.circle(glow_surf, (*self.COLOR_PLAYER, 50), (self.PLAYER_SIZE, self.PLAYER_SIZE), self.PLAYER_SIZE)
        self.screen.blit(glow_surf, (player_rect.centerx - self.PLAYER_SIZE, player_rect.centery - self.PLAYER_SIZE))
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, player_rect, border_radius=2)
        
        # Render UI
        self._render_text(f"Stage: {self.stage}/{self.MAX_STAGES}", (15, 10), self.font_small, self.COLOR_TEXT, self.COLOR_TEXT_SHADOW)
        time_str = f"Time: {self.timer // self.FPS:02d}"
        self._render_text(time_str, (self.SCREEN_WIDTH - 15, 10), self.font_small, self.COLOR_TEXT, self.COLOR_TEXT_SHADOW, align="topright")
        self._render_text(f"Score: {self.score}", (self.SCREEN_WIDTH - 15, self.SCREEN_HEIGHT - 15), self.font_medium, self.COLOR_TEXT, self.COLOR_TEXT_SHADOW, align="bottomright")

        # Render message
        if self.message and self.message_timer > 0:
            self._render_text(self.message, (self.SCREEN_WIDTH/2, self.SCREEN_HEIGHT/2), self.font_large, self.COLOR_TEXT, self.COLOR_TEXT_SHADOW, align="center")
            
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_text(self, text, pos, font, color, shadow_color, align="topleft"):
        text_surf = font.render(text, True, color)
        shadow_surf = font.render(text, True, shadow_color)
        text_rect = text_surf.get_rect()
        
        if align == "topleft":
            text_rect.topleft = pos
        elif align == "topright":
            text_rect.topright = pos
        elif align == "center":
            text_rect.center = pos
        elif align == "bottomright":
            text_rect.bottomright = pos
        
        shadow_rect = text_rect.copy()
        shadow_rect.x += 2
        shadow_rect.y += 2
        
        self.screen.blit(shadow_surf, shadow_rect)
        self.screen.blit(text_surf, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "stage": self.stage,
            "time_left": self.timer / self.FPS
        }

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert obs.dtype == np.uint8
        assert isinstance(info, dict)
        
        # Test observation space from _get_observation
        test_obs = self._get_observation()
        assert test_obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == "__main__":
    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    # --- Human Play Loop ---
    # Use 'space' to jump
    
    total_reward = 0
    
    # Pygame display setup for human play
    pygame.display.set_caption(GameEnv.game_description)
    display_screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    
    while not done:
        # Map keyboard inputs to action space
        keys = pygame.key.get_pressed()
        space_held = keys[pygame.K_SPACE]
        
        action = [0, 1 if space_held else 0, 0] # Movement=None, Space=Held?, Shift=Released
        
        # Handle quit event
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        done = terminated or truncated

        # Render the observation to the display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        display_screen.blit(surf, (0, 0))
        pygame.display.flip()

    print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
    env.close()