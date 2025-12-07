
# Generated: 2025-08-27T14:01:50.091737
# Source Brief: brief_00564.md
# Brief Index: 564

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
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
    user_guide = (
        "Controls: ←→ to move, ↑ to jump. Press space to dash."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A fast-paced pixel art platformer. Jump and dash to collect gems and reach the flag before time runs out!"
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Game constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 60
        self.MAX_STEPS = 30 * self.FPS  # 30 seconds

        # Physics constants
        self.GRAVITY = 0.4
        self.PLAYER_ACCEL = 0.6
        self.PLAYER_FRICTION = -0.12
        self.PLAYER_JUMP_STRENGTH = -10
        self.PLAYER_MAX_SPEED = 6
        self.DASH_SPEED = 12
        self.DASH_DURATION = 8  # frames
        self.DASH_COOLDOWN = 30 # frames

        # Colors
        self.COLOR_BG = (20, 15, 40)
        self.COLOR_PLAYER = (50, 255, 255)
        self.COLOR_PLAYER_TRAIL = (150, 255, 255, 150)
        self.COLOR_PLATFORM_SAFE = (50, 205, 50)
        self.COLOR_PLATFORM_BRITTLE = (220, 20, 60)
        self.COLOR_GEM = (0, 191, 255)
        self.COLOR_FLAGPOLE = (192, 192, 192)
        self.COLOR_FLAG = (255, 0, 0)
        self.COLOR_TEXT = (240, 240, 240)
        
        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 36)
        self.font_small = pygame.font.Font(None, 24)
        
        # Initialize state variables
        self.player_pos = None
        self.player_vel = None
        self.player_facing_right = True
        self.is_grounded = False
        self.is_dashing = 0
        self.dash_cooldown_timer = 0
        self.platforms = []
        self.gems = []
        self.flag_pos = None
        self.particles = []
        self.camera_offset = pygame.Vector2(0, 0)
        
        self.steps = 0
        self.score = 0
        self.time_left = 0
        self.terminated = False
        self.collected_gem_this_step = False
        
        self.reset()
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Reset game state
        self.steps = 0
        self.score = 0
        self.time_left = self.MAX_STEPS
        self.terminated = False
        
        # Reset player
        self.player_pos = pygame.Vector2(100, 200)
        self.player_vel = pygame.Vector2(0, 0)
        self.is_grounded = False
        self.is_dashing = 0
        self.dash_cooldown_timer = 0
        self.player_facing_right = True
        
        # Level Generation
        self._generate_level()
        
        self.particles = []
        self.camera_offset = pygame.Vector2(0, 0)
        
        return self._get_observation(), self._get_info()

    def _generate_level(self):
        self.platforms = []
        self.gems = []
        
        # Ground floor
        self.platforms.append({'rect': pygame.Rect(-200, 350, 1000, 50), 'type': 'safe', 'state': 'solid'})
        
        # Platforms (x, y, w, h, type)
        level_data = [
            (350, 280, 150, 20, 'safe'),
            (600, 220, 150, 20, 'brittle'),
            (850, 180, 100, 20, 'safe'),
            (1100, 250, 120, 20, 'safe'),
            (1300, 200, 50, 20, 'brittle'),
            (1450, 150, 150, 20, 'safe'),
            (1250, 300, 150, 20, 'brittle'),
            (950, 300, 80, 20, 'safe'),
        ]
        for x, y, w, h, p_type in level_data:
            self.platforms.append({'rect': pygame.Rect(x, y, w, h), 'type': p_type, 'state': 'solid', 'timer': 0})

        # Gems (x, y)
        gem_data = [
            (425, 250), (675, 190), (1150, 220), (1325, 170), (1280, 270)
        ]
        self.gems = [pygame.Rect(x, y, 12, 12) for x, y in gem_data]

        # Flag
        self.flag_pos = pygame.Vector2(1525, 130)

    def step(self, action):
        if self.terminated:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        self.time_left -= 1
        self.collected_gem_this_step = False

        self._handle_input(action)
        self._update_physics()
        self._check_interactions()
        self._update_brittle_platforms()
        self._update_particles()
        self._update_camera()

        reward = self._calculate_reward()
        self.terminated = self._check_termination()
        
        return (
            self._get_observation(),
            reward,
            self.terminated,
            False,
            self._get_info()
        )

    def _handle_input(self, action):
        movement, space_held, _ = action
        
        # Horizontal Movement
        if self.is_dashing <= 0:
            if movement == 3:  # Left
                self.player_vel.x += -self.PLAYER_ACCEL
                self.player_facing_right = False
            elif movement == 4:  # Right
                self.player_vel.x += self.PLAYER_ACCEL
                self.player_facing_right = True
        
        # Jumping
        if movement == 1 and self.is_grounded:
            self.player_vel.y = self.PLAYER_JUMP_STRENGTH
            self._add_particles(self.player_pos + pygame.Vector2(0, 15), 10, (200,200,200), 2, 0.5) # Jump dust
            
        # Dashing
        if space_held and self.is_dashing <= 0 and self.dash_cooldown_timer <= 0:
            self.is_dashing = self.DASH_DURATION
            self.dash_cooldown_timer = self.DASH_COOLDOWN
            dash_dir = 1 if self.player_facing_right else -1
            self.player_vel.x = self.DASH_SPEED * dash_dir
            self.player_vel.y = 0 # Horizontal dash

    def _update_physics(self):
        # Update timers
        if self.is_dashing > 0:
            self.is_dashing -= 1
        if self.dash_cooldown_timer > 0:
            self.dash_cooldown_timer -= 1
            
        # Apply friction only when not accelerating and not dashing
        if self.is_dashing <= 0:
            self.player_vel.x += self.player_vel.x * self.PLAYER_FRICTION
            if abs(self.player_vel.x) < 0.1: self.player_vel.x = 0
            self.player_vel.x = max(-self.PLAYER_MAX_SPEED, min(self.PLAYER_MAX_SPEED, self.player_vel.x))

        # Apply gravity
        self.player_vel.y += self.GRAVITY
        if self.player_vel.y > 15: self.player_vel.y = 15 # Terminal velocity
        
        # Move and collide
        player_rect = pygame.Rect(self.player_pos.x, self.player_pos.y, 20, 30)
        
        # Horizontal collision
        self.player_pos.x += self.player_vel.x
        player_rect.x = int(self.player_pos.x)
        for plat in self.platforms:
            if plat['state'] == 'solid' and player_rect.colliderect(plat['rect']):
                if self.player_vel.x > 0:
                    player_rect.right = plat['rect'].left
                elif self.player_vel.x < 0:
                    player_rect.left = plat['rect'].right
                self.player_pos.x = player_rect.x
                self.player_vel.x = 0

        # Vertical collision
        self.player_pos.y += self.player_vel.y
        player_rect.y = int(self.player_pos.y)
        self.is_grounded = False
        for plat in self.platforms:
            if plat['state'] == 'solid' and player_rect.colliderect(plat['rect']):
                if self.player_vel.y > 0:
                    player_rect.bottom = plat['rect'].top
                    self.is_grounded = True
                    self.player_vel.y = 0
                elif self.player_vel.y < 0:
                    player_rect.top = plat['rect'].bottom
                    self.player_vel.y = 0
                self.player_pos.y = player_rect.y

    def _check_interactions(self):
        player_rect = pygame.Rect(self.player_pos.x, self.player_pos.y, 20, 30)
        
        # Gem collection
        for gem in self.gems[:]:
            if player_rect.colliderect(gem):
                self.gems.remove(gem)
                self.score += 10
                self.collected_gem_this_step = True
                self._add_particles(gem.center, 20, self.COLOR_GEM, 3, 1.0) # Gem collection effect
                # sfx: gem_collect.wav

        # Flag reach
        if player_rect.colliderect(pygame.Rect(self.flag_pos.x, self.flag_pos.y, 10, 50)):
            self.score += 100
            self.terminated = True
            # sfx: level_win.wav

        # Fall into pit
        if self.player_pos.y > self.HEIGHT + 50:
            self.terminated = True
            # sfx: fall_death.wav
    
    def _update_brittle_platforms(self):
        player_rect = pygame.Rect(self.player_pos.x, self.player_pos.y, 20, 30)
        for plat in self.platforms:
            if plat['type'] == 'brittle' and plat['state'] == 'solid':
                # Check if player is standing on it
                on_top_rect = pygame.Rect(player_rect.x, player_rect.bottom, player_rect.width, 2)
                if on_top_rect.colliderect(plat['rect']):
                    plat['timer'] += 1
                    if plat['timer'] > self.FPS * 0.5: # 0.5 second to crumble
                        plat['state'] = 'crumbling'
                        plat['timer'] = self.FPS * 0.3 # Crumble animation duration
                else:
                    plat['timer'] = 0 # Reset timer if player leaves
            elif plat['state'] == 'crumbling':
                plat['timer'] -= 1
                if plat['timer'] <= 0:
                    plat['state'] = 'gone'
                    self._add_particles(plat['rect'].center, 30, self.COLOR_PLATFORM_BRITTLE, 4, 0.2) # Crumble effect

    def _update_particles(self):
        # Dash trail
        if self.is_dashing > 0:
            self._add_particles(self.player_pos + pygame.Vector2(10, 15), 1, self.COLOR_PLAYER_TRAIL, 5, 0, fixed_lifetime=15)
        
        for p in self.particles[:]:
            p['pos'] += p['vel']
            p['lifetime'] -= 1
            p['radius'] -= 0.1
            if p['lifetime'] <= 0 or p['radius'] <= 0:
                self.particles.remove(p)

    def _update_camera(self):
        target_x = self.player_pos.x - self.WIDTH / 2 + 10 # Center player
        target_y = self.player_pos.y - self.HEIGHT * 2 / 3 # Keep player in lower 2/3
        
        # Clamp camera to not show empty space below level
        target_y = min(target_y, 20)
        
        # Smooth camera movement
        self.camera_offset.x += (target_x - self.camera_offset.x) * 0.1
        self.camera_offset.y += (target_y - self.camera_offset.y) * 0.1

    def _calculate_reward(self):
        reward = 0.0
        # Living reward
        reward += 0.01 
        
        # Event-based rewards
        if self.collected_gem_this_step:
            reward += 10
        if self.terminated and self.player_pos.y < self.HEIGHT + 50 and self.time_left > 0: # Reached flag
            reward += 100

        # Proximity penalty
        is_near_pit = True
        player_feet = self.player_pos.y + 30
        for plat in self.platforms:
            if plat['rect'].left < self.player_pos.x < plat['rect'].right and plat['rect'].top > player_feet:
                 if plat['rect'].top - player_feet < 150: # Check for platform below
                    is_near_pit = False
                    break
        if is_near_pit and self.player_pos.y > 200: # Only penalize when high enough to be over a pit
            reward -= 0.5

        return reward

    def _check_termination(self):
        return self.terminated or self.time_left <= 0 or self.steps >= self.MAX_STEPS

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        cam_x, cam_y = int(self.camera_offset.x), int(self.camera_offset.y)

        # Render background elements (parallax)
        for i in range(50):
            x = (hash(i*10) % (self.WIDTH * 2)) - self.WIDTH/2
            y = (hash(i*20) % self.HEIGHT)
            # Far stars
            px = int(x - cam_x * 0.1) % self.WIDTH
            py = int(y - cam_y * 0.1) % self.HEIGHT
            pygame.draw.rect(self.screen, (60, 60, 100), (px, py, 1, 1))
            # Near stars
            px = int(x - cam_x * 0.3) % self.WIDTH
            py = int(y - cam_y * 0.3) % self.HEIGHT
            pygame.draw.rect(self.screen, (120, 120, 180), (px, py, 2, 2))

        # Render particles
        for p in self.particles:
            pos = (int(p['pos'].x - cam_x), int(p['pos'].y - cam_y))
            color = p['color']
            if len(color) == 4: # Handle alpha
                pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], int(p['radius']), color)
            else:
                pygame.draw.circle(self.screen, color, pos, int(p['radius']))

        # Render platforms
        for plat in self.platforms:
            if plat['state'] == 'gone': continue
            color = self.COLOR_PLATFORM_SAFE if plat['type'] == 'safe' else self.COLOR_PLATFORM_BRITTLE
            if plat['state'] == 'crumbling':
                if self.steps % 10 < 5: color = self.COLOR_BG # Flicker effect
            
            rect = plat['rect'].move(-cam_x, -cam_y)
            pygame.draw.rect(self.screen, color, rect, border_radius=3)
            pygame.draw.rect(self.screen, tuple(max(0, c-40) for c in color[:3]), rect, width=2, border_radius=3)


        # Render gems
        for gem in self.gems:
            pulse = (math.sin(self.steps * 0.2) + 1) / 2 * 3
            rect = gem.move(-cam_x, -cam_y)
            pygame.draw.rect(self.screen, self.COLOR_GEM, rect.inflate(pulse, pulse), border_radius=4)
            pygame.draw.rect(self.screen, (200, 255, 255), rect.inflate(pulse-4, pulse-4), border_radius=2)

        # Render flag
        pole_rect = pygame.Rect(self.flag_pos.x - cam_x, self.flag_pos.y - cam_y, 5, 50)
        flag_points = [
            (self.flag_pos.x + 5 - cam_x, self.flag_pos.y - cam_y),
            (self.flag_pos.x + 35 - cam_x, self.flag_pos.y + 10 - cam_y),
            (self.flag_pos.x + 5 - cam_x, self.flag_pos.y + 20 - cam_y)
        ]
        pygame.draw.rect(self.screen, self.COLOR_FLAGPOLE, pole_rect)
        pygame.draw.polygon(self.screen, self.COLOR_FLAG, flag_points)

        # Render player
        player_rect = pygame.Rect(0, 0, 20, 30)
        player_rect.center = (self.player_pos.x + 10 - cam_x, self.player_pos.y + 15 - cam_y)
        
        # Squash and stretch
        if not self.is_grounded:
            player_rect.height = 32
            player_rect.width = 18
        if self.is_dashing > 0:
            player_rect.width = 25
            player_rect.height = 25
        player_rect.center = (self.player_pos.x + 10 - cam_x, self.player_pos.y + 15 - cam_y)

        pygame.draw.rect(self.screen, self.COLOR_PLAYER, player_rect, border_radius=4)
        pygame.draw.rect(self.screen, (200, 255, 255), player_rect.inflate(-4,-4), border_radius=4)


    def _render_ui(self):
        score_text = self.font.render(f"Score: {self.score}", True, self.COLOR_TEXT)
        time_text = self.font.render(f"Time: {self.time_left / self.FPS:.1f}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))
        self.screen.blit(time_text, (10, 40))

        if self.dash_cooldown_timer > 0:
            dash_ready_text = self.font_small.render("Dash Cooldown", True, (255,100,100))
            self.screen.blit(dash_ready_text, (self.WIDTH - 120, 10))
        else:
            dash_ready_text = self.font_small.render("Dash Ready", True, (100,255,100))
            self.screen.blit(dash_ready_text, (self.WIDTH - 110, 10))


    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "time_left": self.time_left,
            "player_pos": (self.player_pos.x, self.player_pos.y),
            "terminated": self.terminated
        }

    def _add_particles(self, pos, count, color, max_radius, speed_mult, fixed_lifetime=None):
        for _ in range(count):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1, 3) * speed_mult
            vel = pygame.Vector2(math.cos(angle), math.sin(angle)) * speed
            self.particles.append({
                'pos': pygame.Vector2(pos),
                'vel': vel,
                'radius': random.uniform(1, max_radius),
                'color': color,
                'lifetime': fixed_lifetime if fixed_lifetime is not None else random.randint(20, 40)
            })

    def close(self):
        pygame.font.quit()
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
    # This block allows you to play the game directly
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Pixel Platformer")
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    while running:
        movement = 0 # no-op
        space_held = 0
        shift_held = 0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            movement = 3
        elif keys[pygame.K_RIGHT]:
            movement = 4
        
        if keys[pygame.K_UP]:
            movement = 1
        elif keys[pygame.K_DOWN]:
            movement = 2

        if keys[pygame.K_SPACE]:
            space_held = 1
        
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            shift_held = 1
            
        action = [movement, space_held, shift_held]
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # Draw the observation from the environment to the screen
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        
        if terminated or truncated:
            print(f"Episode finished. Total Reward: {total_reward:.2f}, Score: {info['score']}")
            total_reward = 0
            obs, info = env.reset()
            # Add a small delay to notice the reset
            pygame.time.wait(1000)

        clock.tick(env.FPS)
        
    env.close()