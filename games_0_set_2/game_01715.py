
# Generated: 2025-08-28T02:28:33.098947
# Source Brief: brief_01715.md
# Brief Index: 1715

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
from collections import deque
import os
import pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Short, user-facing control string
    user_guide = (
        "Controls: ←→ to move. ↑ for a normal jump, Space for a high jump. "
        "Shift to dash. ↓ to fall faster."
    )

    # Short, user-facing description of the game
    game_description = (
        "Hop between scrolling platforms, dodge deadly obstacles, and race the clock to "
        "reach the goal in this fast-paced arcade jumper."
    )

    # Frames auto-advance at 30fps
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Screen and rendering setup
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 30
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.Font(None, 24)
        self.font_large = pygame.font.Font(None, 50)
        self.render_mode = render_mode

        # Gymnasium spaces
        self.observation_space = Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Colors
        self.COLOR_BG_TOP = (10, 20, 40)
        self.COLOR_BG_BOTTOM = (0, 0, 0)
        self.COLOR_PLAYER = (255, 255, 0)
        self.COLOR_PLAYER_GLOW = (255, 255, 150, 100)
        self.COLOR_PLATFORM = (100, 200, 255)
        self.COLOR_PLATFORM_OUTLINE = (200, 255, 255)
        self.COLOR_OBSTACLE = (255, 50, 50)
        self.COLOR_OBSTACLE_GLOW = (255, 150, 150, 100)
        self.COLOR_TEXT = (255, 255, 255)
        self.COLOR_PARTICLE_JUMP = (200, 255, 255)
        self.COLOR_PARTICLE_LAND = (255, 255, 255)
        self.COLOR_PARTICLE_DASH = (255, 255, 0)

        # Game constants
        self.MAX_STEPS = 60 * self.FPS  # 60 seconds
        self.GRAVITY = 0.6
        self.PLAYER_SIZE = 16
        self.PLAYER_MOVE_SPEED = 4
        self.JUMP_VELOCITY = -10
        self.BIG_JUMP_VELOCITY = -13
        self.DASH_SPEED = 8
        self.FAST_FALL_SPEED = 3
        self.PLATFORM_HEIGHT = 12
        self.TARGET_PLATFORM = 10

        # Pre-render background
        self.bg_surface = self._create_gradient_background()

        # Initialize state variables
        self.player_pos = None
        self.player_vel = None
        self.on_ground = None
        self.can_dash = None
        self.platforms = None
        self.obstacles = None
        self.particles = None
        self.steps = None
        self.score = None
        self.game_over = None
        self.highest_platform = None
        self.scroll_speed = None
        self.obstacle_density = None
        
        # This will be called again by the environment wrapper, but is good practice.
        self.reset()
        
        # Run validation check
        self.validate_implementation()


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.player_pos = np.array([self.WIDTH / 2, self.HEIGHT * 0.75], dtype=float)
        self.player_vel = np.array([0.0, 0.0], dtype=float)
        self.on_ground = True
        self.can_dash = True

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.highest_platform = 0

        self.scroll_speed = 1.0
        self.obstacle_density = 0.1

        self.particles = deque()
        self.platforms = []
        self.obstacles = []
        
        # Generate initial platforms
        initial_platform = pygame.Rect(
            self.player_pos[0] - 50, self.player_pos[1] + self.PLAYER_SIZE, 100, self.PLATFORM_HEIGHT
        )
        self.platforms.append(initial_platform)
        
        for i in range(1, 15):
             self._generate_platform(is_initial=True)

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        reward = 0
        self.steps += 1

        # Handle player actions
        self._handle_input(action)
        reward -= 0.02 if action[0] == 0 else 0

        # Update game state
        self._update_player_physics()
        self._update_world()
        
        # Check for collisions and update state based on them
        landed_on_new_platform, on_platform_index = self._check_platform_collisions()
        if landed_on_new_platform > self.highest_platform:
            self.highest_platform = landed_on_new_platform
            reward += 1 # Reached a new platform
            # sound: new_platform_reached.wav
        
        if self.on_ground:
            reward += 0.1 # Survived another step on a platform

        # Check for termination conditions
        terminated = self._check_termination()
        
        # Calculate terminal rewards
        if terminated:
            if self.player_pos[1] > self.HEIGHT:
                reward = -100 # Fell off screen
                # sound: fall_off.wav
            elif self._check_obstacle_collisions():
                reward = -5 # Hit an obstacle
            elif self.steps >= self.MAX_STEPS:
                reward = -50 # Time ran out
            elif self.highest_platform >= self.TARGET_PLATFORM:
                reward = 50 # Reached the goal
                # sound: victory.wav
        
        self.score += reward
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated is always False
            self._get_info()
        )

    def _handle_input(self, action):
        movement, space_btn, shift_btn = action[0], action[1] == 1, action[2] == 1

        # Horizontal movement
        if movement == 3:  # Left
            self.player_vel[0] = -self.PLAYER_MOVE_SPEED
        elif movement == 4:  # Right
            self.player_vel[0] = self.PLAYER_MOVE_SPEED
        else:
            # Dampen horizontal velocity if not moving and in air
            if not self.on_ground:
                self.player_vel[0] *= 0.9
            else:
                self.player_vel[0] = 0

        # Vertical actions
        if self.on_ground:
            if movement == 1:  # Up (Normal Jump)
                self.player_vel[1] = self.JUMP_VELOCITY
                self._create_particles(10, self.COLOR_PARTICLE_JUMP, -3)
                # sound: jump.wav
            elif space_btn:  # Space (Big Jump)
                self.player_vel[1] = self.BIG_JUMP_VELOCITY
                self._create_particles(20, self.COLOR_PARTICLE_JUMP, -5)
                # sound: big_jump.wav
        
        # Fast fall
        if not self.on_ground and movement == 2: # Down
            if self.player_vel[1] < self.FAST_FALL_SPEED:
                 self.player_vel[1] = self.FAST_FALL_SPEED

        # Dashing
        if shift_btn and self.can_dash:
            self.can_dash = False
            # Dash direction based on current velocity or last input
            dash_dir = np.sign(self.player_vel[0]) if self.player_vel[0] != 0 else 1
            self.player_vel[0] = self.DASH_SPEED * dash_dir
            self.player_vel[1] = -2 # Small upward boost for feel
            self._create_particles(30, self.COLOR_PARTICLE_DASH, 0, trail=True)
            # sound: dash.wav

    def _update_player_physics(self):
        # Apply gravity
        if not self.on_ground:
            self.player_vel[1] += self.GRAVITY

        # Update position
        self.player_pos += self.player_vel

        # Screen wrap horizontal
        if self.player_pos[0] < 0:
            self.player_pos[0] = self.WIDTH
        elif self.player_pos[0] > self.WIDTH:
            self.player_pos[0] = 0
            
    def _check_platform_collisions(self):
        player_rect = pygame.Rect(self.player_pos[0], self.player_pos[1], self.PLAYER_SIZE, self.PLAYER_SIZE)
        
        was_on_ground = self.on_ground
        self.on_ground = False
        landed_platform_index = -1

        for i, plat in enumerate(self.platforms):
            # Check for landing on a platform
            if (self.player_vel[1] >= 0 and
                player_rect.colliderect(plat) and
                abs((player_rect.bottom) - plat.top) < self.PLAYER_SIZE):
                
                self.player_pos[1] = plat.top - self.PLAYER_SIZE
                self.player_vel[1] = self.scroll_speed # Stick to platform
                self.on_ground = True
                self.can_dash = True
                landed_platform_index = i
                
                if not was_on_ground: # Just landed
                    self._create_particles(15, self.COLOR_PARTICLE_LAND, 1)
                    # sound: land.wav
                break
        
        return landed_platform_index, landed_platform_index != -1

    def _check_obstacle_collisions(self):
        player_rect = pygame.Rect(self.player_pos[0], self.player_pos[1], self.PLAYER_SIZE, self.PLAYER_SIZE)
        for obs in self.obstacles:
            # Simple circle-rect collision
            if player_rect.clipline(obs['pos'], (obs['pos'][0], obs['pos'][1] + 1)):
                return True
            if pygame.math.Vector2(player_rect.center).distance_to(obs['pos']) < obs['radius'] + self.PLAYER_SIZE / 2:
                self._create_particles(50, self.COLOR_OBSTACLE, 0, explosion=True)
                # sound: obstacle_hit.wav
                return True
        return False

    def _update_world(self):
        # Scroll platforms and obstacles
        for plat in self.platforms:
            plat.y += self.scroll_speed
        for obs in self.obstacles:
            obs['pos'][1] += self.scroll_speed
        
        # If player is on ground, they scroll too
        if self.on_ground:
            self.player_pos[1] += self.scroll_speed

        # Remove off-screen elements
        self.platforms = [p for p in self.platforms if p.top < self.HEIGHT]
        self.obstacles = [o for o in self.obstacles if o['pos'][1] < self.HEIGHT + o['radius']]

        # Generate new platforms if needed
        while len(self.platforms) < 15:
            self._generate_platform()
            
        # Update difficulty
        if self.steps > 0:
            if self.steps % 200 == 0:
                self.scroll_speed = min(3.0, self.scroll_speed + 0.01)
            if self.steps % 300 == 0:
                self.obstacle_density = min(0.5, self.obstacle_density + 0.05)

    def _generate_platform(self, is_initial=False):
        last_platform = self.platforms[-1]
        
        width = self.np_random.integers(60, 150)
        
        # Position new platform relative to the last one
        min_y = last_platform.top - self.np_random.integers(60, 120)
        max_x_offset = 150
        min_x = max(20, last_platform.centerx - max_x_offset)
        max_x = min(self.WIDTH - width - 20, last_platform.centerx + max_x_offset)
        
        # Ensure x range is valid
        if min_x >= max_x:
            min_x = 20
            max_x = self.WIDTH - width - 20
        
        x = self.np_random.integers(min_x, max_x)
        y = min_y if not is_initial else last_platform.y + self.np_random.integers(60, 80)
        
        new_platform = pygame.Rect(x, y, width, self.PLATFORM_HEIGHT)
        self.platforms.append(new_platform)
        
        # Potentially add an obstacle
        if self.np_random.random() < self.obstacle_density and not is_initial:
            obs_radius = self.np_random.integers(8, 12)
            obs_x = new_platform.x + self.np_random.integers(obs_radius, new_platform.width - obs_radius)
            obs_y = new_platform.top - obs_radius
            self.obstacles.append({'pos': [obs_x, obs_y], 'radius': obs_radius})

    def _check_termination(self):
        if self.game_over: return True
        
        if (self.player_pos[1] > self.HEIGHT or
            self.steps >= self.MAX_STEPS or
            self._check_obstacle_collisions() or
            self.highest_platform >= self.TARGET_PLATFORM):
            self.game_over = True
            return True
        return False

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "highest_platform": self.highest_platform,
            "time_left": (self.MAX_STEPS - self.steps) / self.FPS
        }

    def _get_observation(self):
        self._render_game()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Background
        self.screen.blit(self.bg_surface, (0, 0))

        # Render elements
        self._render_platforms_obstacles()
        self._render_particles()
        self._render_player()
        self._render_ui()

    def _create_gradient_background(self):
        bg = pygame.Surface((self.WIDTH, self.HEIGHT))
        for y in range(self.HEIGHT):
            color = [
                self.COLOR_BG_TOP[i] + (self.COLOR_BG_BOTTOM[i] - self.COLOR_BG_TOP[i]) * (y / self.HEIGHT)
                for i in range(3)
            ]
            pygame.draw.line(bg, color, (0, y), (self.WIDTH, y))
        return bg

    def _render_platforms_obstacles(self):
        for plat in self.platforms:
            pygame.draw.rect(self.screen, self.COLOR_PLATFORM, plat, border_radius=3)
            pygame.draw.rect(self.screen, self.COLOR_PLATFORM_OUTLINE, plat, 1, border_radius=3)
        
        for obs in self.obstacles:
            pos = (int(obs['pos'][0]), int(obs['pos'][1]))
            radius = int(obs['radius'])
            # Glow effect
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], radius + 3, self.COLOR_OBSTACLE_GLOW)
            # Main circle
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], radius, self.COLOR_OBSTACLE)
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], radius, self.COLOR_OBSTACLE)

    def _render_player(self):
        player_rect = pygame.Rect(int(self.player_pos[0]), int(self.player_pos[1]), self.PLAYER_SIZE, self.PLAYER_SIZE)
        
        # Glow effect
        glow_surface = pygame.Surface((self.PLAYER_SIZE * 2, self.PLAYER_SIZE * 2), pygame.SRCALPHA)
        pygame.draw.rect(glow_surface, self.COLOR_PLAYER_GLOW, glow_surface.get_rect(), border_radius=8)
        self.screen.blit(glow_surface, (player_rect.x - self.PLAYER_SIZE/2, player_rect.y - self.PLAYER_SIZE/2))
        
        # Player
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, player_rect, border_radius=3)

    def _render_particles(self):
        # Update and draw particles
        for _ in range(len(self.particles)):
            p = self.particles.popleft()
            p['pos'] += p['vel']
            p['life'] -= 1
            if p['life'] > 0:
                alpha = max(0, 255 * (p['life'] / p['max_life']))
                p['color'] = (*p['base_color'], alpha)
                
                size = max(1, int(p['size'] * (p['life'] / p['max_life'])))
                pygame.draw.circle(self.screen, p['color'], [int(p['pos'][0]), int(p['pos'][1])], size)
                self.particles.append(p)
    
    def _create_particles(self, count, color, y_vel_mod, trail=False, explosion=False):
        for _ in range(count):
            if trail:
                vel = np.array([-self.player_vel[0] * 0.2, self.np_random.uniform(-0.5, 0.5)])
                pos = self.player_pos + np.array([self.PLAYER_SIZE/2, self.PLAYER_SIZE/2])
            elif explosion:
                angle = self.np_random.uniform(0, 2 * math.pi)
                speed = self.np_random.uniform(1, 6)
                vel = np.array([math.cos(angle) * speed, math.sin(angle) * speed])
                pos = self.player_pos + np.array([self.PLAYER_SIZE/2, self.PLAYER_SIZE/2])
            else: # Jump/Land
                vel = np.array([self.np_random.uniform(-1.5, 1.5), self.np_random.uniform(0, 2) + y_vel_mod])
                pos = self.player_pos + np.array([self.PLAYER_SIZE/2, self.PLAYER_SIZE])
                
            life = self.np_random.integers(10, 25)
            self.particles.append({
                'pos': pos.copy(),
                'vel': vel,
                'life': life,
                'max_life': life,
                'size': self.np_random.integers(2, 5),
                'base_color': color,
                'color': (*color, 255)
            })

    def _render_ui(self):
        # Platform progress
        progress_text = f"Platform: {self.highest_platform} / {self.TARGET_PLATFORM}"
        text_surf = self.font_small.render(progress_text, True, self.COLOR_TEXT)
        self.screen.blit(text_surf, (10, 10))

        # Time remaining
        time_left = max(0, (self.MAX_STEPS - self.steps) / self.FPS)
        time_text = f"Time: {time_left:.1f}"
        time_color = self.COLOR_OBSTACLE if time_left < 10 else self.COLOR_TEXT
        text_surf = self.font_small.render(time_text, True, time_color)
        self.screen.blit(text_surf, (self.WIDTH - text_surf.get_width() - 10, 10))
        
        # Score
        score_text = f"Score: {self.score:.2f}"
        text_surf = self.font_small.render(score_text, True, self.COLOR_TEXT)
        self.screen.blit(text_surf, (10, 35))

        # Game Over / Win message
        if self.game_over:
            if self.highest_platform >= self.TARGET_PLATFORM:
                msg = "YOU WIN!"
                color = self.COLOR_PLAYER
            else:
                msg = "GAME OVER"
                color = self.COLOR_OBSTACLE
            
            text_surf = self.font_large.render(msg, True, color)
            text_rect = text_surf.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
            self.screen.blit(text_surf, text_rect)

    def close(self):
        pygame.quit()
        
    def validate_implementation(self):
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
    # To play the game manually
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # Pygame setup for manual play
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Platform Hopper")
    clock = pygame.time.Clock()

    terminated = False
    running = True
    while running:
        if terminated:
            # Show game over for a bit then reset
            pygame.time.wait(2000)
            obs, info = env.reset()
            terminated = False

        # --- Action mapping for manual play ---
        keys = pygame.key.get_pressed()
        action = [0, 0, 0] # [movement, space, shift]
        
        if keys[pygame.K_UP]:
            action[0] = 1
        elif keys[pygame.K_DOWN]:
            action[0] = 2
        elif keys[pygame.K_LEFT]:
            action[0] = 3
        elif keys[pygame.K_RIGHT]:
            action[0] = 4
        
        if keys[pygame.K_SPACE]:
            action[1] = 1
        
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            action[2] = 1
        
        # --- Gym step ---
        obs, reward, terminated, truncated, info = env.step(action)

        # --- Rendering ---
        # The observation is already a rendered frame, we just need to display it
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        # --- Event handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        clock.tick(env.FPS)
        
    env.close()