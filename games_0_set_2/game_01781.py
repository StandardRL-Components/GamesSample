import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
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

    user_guide = (
        "Controls: Use arrow keys to move. Press Space to jump and Shift to activate your shield."
    )

    game_description = (
        "Escape a hostile alien planet by dodging obstacles and laser fire to reach your spaceship."
    )

    auto_advance = True

    # --- Constants ---
    # Screen
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    
    # Colors
    COLOR_BG = (20, 10, 30)
    COLOR_STARS = [(50, 40, 60), (80, 70, 90)]
    COLOR_PLAYER = (0, 255, 128)
    COLOR_PLAYER_DMG = (255, 100, 100)
    COLOR_OBSTACLE = (100, 110, 120)
    COLOR_LASER_CHARGE = (255, 150, 0)
    COLOR_LASER_FIRE = (255, 50, 50)
    COLOR_SHIELD = (0, 180, 255)
    COLOR_SPACESHIP = (200, 200, 220)
    COLOR_UI_TEXT = (220, 220, 240)
    COLOR_HEALTH_BAR_BG = (80, 20, 20)
    COLOR_HEALTH_BAR_FG = (200, 40, 40)
    
    # Game parameters
    MAX_STEPS = 5000
    LEVEL_LENGTH = MAX_STEPS * 5 # Logical length of the level
    GROUND_Y = 350
    
    # Player
    PLAYER_SIZE = (20, 40)
    PLAYER_ACCEL = 0.8
    PLAYER_FRICTION = -0.15
    PLAYER_GRAVITY = 0.6
    PLAYER_JUMP_STRENGTH = -12
    PLAYER_MAX_SPEED_X = 6
    PLAYER_MAX_SPEED_Y = 15
    
    # Shield
    SHIELD_DURATION = 60  # 2 seconds at 30fps
    SHIELD_COOLDOWN = 120 # 4 seconds
    SHIELD_RADIUS = 40

    # Obstacles
    OBSTACLE_MIN_H = 20
    OBSTACLE_MAX_H = 80
    OBSTACLE_MIN_W = 20
    OBSTACLE_MAX_W = 60
    
    # Lasers
    LASER_HEIGHT = 4
    LASER_CHARGE_TIME = 45 # 1.5s
    LASER_FIRE_TIME = 15   # 0.5s

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        self.observation_space = Box(low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8)
        self.action_space = MultiDiscrete([5, 2, 2])

        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.Font(None, 24)
        self.font_large = pygame.font.Font(None, 48)

        self.np_random = None
        self.player_pos = pygame.Vector2(0, 0)
        self.player_vel = pygame.Vector2(0, 0)
        self.player_health = 100.0
        self.on_ground = False
        
        self.is_shielding = False
        self.shield_timer = 0
        self.shield_cooldown_timer = 0
        self.shield_just_activated = False

        self.obstacles = []
        self.lasers = []
        self.particles = []
        self.stars = []
        
        self.world_scroll_x = 0.0
        self.base_scroll_speed = 3.0
        self.game_speed = 1.0
        
        self.steps = 0
        self.score = 0.0
        self.game_over = False
        
        self.last_space_held = False
        self.last_shift_held = False

        self.screen_shake_timer = 0

        self.spaceship_rect = pygame.Rect(0,0,0,0)

        # self.reset() is called by the wrapper, no need to call it here.
        
        # This can be commented out for performance but is good for development
        # self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.np_random = np.random.default_rng(seed)
        else:
            # Use a default seed if none is provided to ensure reproducibility
            # in case the user doesn't seed the environment.
            self.np_random = np.random.default_rng()


        self.player_pos = pygame.Vector2(self.SCREEN_WIDTH * 0.2, self.GROUND_Y - self.PLAYER_SIZE[1])
        self.player_vel = pygame.Vector2(0, 0)
        self.player_health = 100.0
        self.on_ground = True
        
        self.is_shielding = False
        self.shield_timer = 0
        self.shield_cooldown_timer = 0
        self.shield_just_activated = False
        
        self.obstacles = []
        self.lasers = []
        self.particles = []
        
        self.world_scroll_x = 0.0
        self.game_speed = 1.0
        self.base_laser_freq = 0.01

        self.steps = 0
        self.score = 0.0
        self.game_over = False
        
        self.last_space_held = False
        self.last_shift_held = False

        self.screen_shake_timer = 0

        self.spaceship_rect = pygame.Rect(self.LEVEL_LENGTH + 100, self.GROUND_Y - 150, 100, 150)
        
        self.stars = [
            (self.np_random.integers(0, self.SCREEN_WIDTH), self.np_random.integers(0, self.SCREEN_HEIGHT), self.np_random.uniform(0.1, 0.5))
            for _ in range(150)
        ]

        # Pre-spawn some initial obstacles
        for i in range(5):
            self._spawn_obstacle(spawn_x=self.SCREEN_WIDTH + i * 200)

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        self.clock.tick(30)
        reward = 0.0
        
        # --- 1. Handle Input & Actions ---
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        # Movement
        if movement == 1: # Up
            # This is not a standard platformer control, but we keep it as per game logic.
            # In a typical platformer, only jump affects Y velocity upwards.
            # self.player_vel.y -= self.PLAYER_ACCEL * 1.5 
            pass # No-op for 'up' unless jumping
        if movement == 2: # Down
            # This is also not standard.
            # self.player_vel.y += self.PLAYER_ACCEL * 1.5
            pass # No-op for 'down'
        if movement == 3: # Left
            self.player_vel.x -= self.PLAYER_ACCEL
            reward -= 0.05 # Penalty for moving away from goal
        if movement == 4: # Right
            self.player_vel.x += self.PLAYER_ACCEL
            reward += 0.02 # Reward for moving towards goal
        
        # Jump (rising edge detection)
        if space_held and not self.last_space_held and self.on_ground:
            self.player_vel.y = self.PLAYER_JUMP_STRENGTH
            self.on_ground = False
            # Sound: Jump
            
        # Shield (rising edge detection)
        self.shield_just_activated = False
        if shift_held and not self.last_shift_held and self.shield_cooldown_timer <= 0:
            self.is_shielding = True
            self.shield_timer = self.SHIELD_DURATION
            self.shield_cooldown_timer = self.SHIELD_COOLDOWN
            self.shield_just_activated = True
            # Sound: Shield Activate
            
        self.last_space_held = space_held
        self.last_shift_held = shift_held

        # --- 2. Update Game State ---
        self._update_player()
        self._update_world()
        self._update_obstacles()
        self._update_lasers()
        self._update_particles()
        
        # --- 3. Handle Collisions & Events ---
        reward += self._handle_collisions()

        # --- 4. Termination & Score ---
        self.steps += 1
        terminated = False
        truncated = False
        
        if self.player_health <= 0:
            self.player_health = 0
            terminated = True
            reward -= 100.0
            self.game_over = True
            # Sound: Player Death
        
        if self.steps >= self.MAX_STEPS:
            truncated = True
            self.game_over = True

        player_rect = self._get_player_rect()
        if player_rect.colliderect(self.spaceship_rect):
            terminated = True
            self.game_over = True
            if self.player_health >= 75:
                reward += 100.0
                # Sound: Win
            else:
                reward += 20.0 # Reached but not in good condition
                # Sound: Partial Win
        
        if not (movement == 3 or movement == 4):
             reward -= 0.01 # Small penalty for standing still

        self.score += reward

        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _update_player(self):
        # Apply friction
        self.player_vel.x += self.player_vel.x * self.PLAYER_FRICTION
        if abs(self.player_vel.x) < 0.1: self.player_vel.x = 0
        
        # Apply gravity if not on ground
        if not self.on_ground:
            self.player_vel.y += self.PLAYER_GRAVITY
            
        # Clamp velocity
        self.player_vel.x = max(-self.PLAYER_MAX_SPEED_X, min(self.PLAYER_MAX_SPEED_X, self.player_vel.x))
        self.player_vel.y = max(-self.PLAYER_MAX_SPEED_Y, min(self.PLAYER_MAX_SPEED_Y, self.player_vel.y))

        # Update position
        self.player_pos += self.player_vel
        
        # Ground collision
        if self.player_pos.y + self.PLAYER_SIZE[1] >= self.GROUND_Y:
            self.player_pos.y = self.GROUND_Y - self.PLAYER_SIZE[1]
            self.player_vel.y = 0
            self.on_ground = True
            
        # Screen bounds
        self.player_pos.x = max(0, min(self.SCREEN_WIDTH - self.PLAYER_SIZE[0], self.player_pos.x))
        if self.player_pos.y < 0:
            self.player_pos.y = 0
            if self.player_vel.y < 0:
                self.player_vel.y = 0

        # Shield logic
        if self.is_shielding:
            self.shield_timer -= 1
            if self.shield_timer <= 0:
                self.is_shielding = False
                # Sound: Shield Deactivate
        if self.shield_cooldown_timer > 0:
            self.shield_cooldown_timer -= 1

    def _update_world(self):
        # Increase difficulty
        if self.steps > 0:
            if self.steps % 250 == 0:
                self.game_speed = min(3.0, self.game_speed + 0.05)
            if self.steps % 500 == 0:
                self.base_laser_freq = min(0.05, self.base_laser_freq * 1.01)

        scroll_amount = self.base_scroll_speed * self.game_speed
        self.world_scroll_x += scroll_amount
        
        # Update spaceship position
        self.spaceship_rect.x = self.LEVEL_LENGTH + 100 - self.world_scroll_x

    def _update_obstacles(self):
        scroll_amount = self.base_scroll_speed * self.game_speed
        for obs in self.obstacles:
            obs['rect'].x -= scroll_amount
        
        self.obstacles = [obs for obs in self.obstacles if obs['rect'].right > 0]
        
        if not self.obstacles or self.obstacles[-1]['rect'].right < self.SCREEN_WIDTH - self.np_random.integers(150, 300):
            self._spawn_obstacle()
            
    def _spawn_obstacle(self, spawn_x=None):
        w = self.np_random.integers(self.OBSTACLE_MIN_W, self.OBSTACLE_MAX_W)
        h = self.np_random.integers(self.OBSTACLE_MIN_H, self.OBSTACLE_MAX_H)
        x = spawn_x if spawn_x is not None else self.SCREEN_WIDTH + 50
        y = self.GROUND_Y - h
        self.obstacles.append({'rect': pygame.Rect(x, y, w, h), 'dodged': False})

    def _update_lasers(self):
        scroll_amount = self.base_scroll_speed * self.game_speed
        for laser in self.lasers:
            laser['pos'].x -= scroll_amount
            if laser['state'] == 'charging':
                laser['timer'] -= 1
                if laser['timer'] <= 0:
                    laser['state'] = 'firing'
                    laser['timer'] = self.LASER_FIRE_TIME
                    # Sound: Laser Fire
            elif laser['state'] == 'firing':
                laser['timer'] -= 1
                if laser['timer'] <= 0:
                    laser['state'] = 'off'
        
        self.lasers = [l for l in self.lasers if l['pos'].x + self.SCREEN_WIDTH > 0 and l['state'] != 'off']
        
        if self.np_random.random() < self.base_laser_freq * self.game_speed:
            self._spawn_laser()

    def _spawn_laser(self):
        y = self.np_random.integers(50, self.GROUND_Y - 50)
        self.lasers.append({
            'pos': pygame.Vector2(self.SCREEN_WIDTH, y),
            'state': 'charging',
            'timer': self.LASER_CHARGE_TIME,
            'dodged': False
        })
        # Sound: Laser Charge

    def _update_particles(self):
        for p in self.particles:
            p['pos'] += p['vel']
            p['life'] -= 1
        self.particles = [p for p in self.particles if p['life'] > 0]

    def _handle_collisions(self):
        reward = 0.0
        player_rect = self._get_player_rect()

        # Obstacles
        for obs in self.obstacles:
            if player_rect.colliderect(obs['rect']):
                self.player_health -= 25
                reward -= 5.0
                self.player_vel.x *= -0.5 # Bounce off
                self.screen_shake_timer = 10
                self._create_particles(player_rect.center, 20, self.COLOR_PLAYER_DMG, 3, 7)
                obs['rect'].right = -1 # Mark for removal
                # Sound: Player Hit Obstacle
            elif not obs['dodged'] and obs['rect'].right < player_rect.left:
                reward += 2.0
                obs['dodged'] = True
        
        # Lasers
        for laser in self.lasers:
            if laser['state'] == 'firing':
                laser_line_y = laser['pos'].y
                if player_rect.top < laser_line_y < player_rect.bottom:
                    if self.is_shielding:
                        if not laser['dodged']:
                            # Perfect block reward
                            if self.shield_timer > self.SHIELD_DURATION - 10:
                                reward += 5.0
                            else:
                                reward += 1.0 # Normal block
                            self._create_particles( (player_rect.centerx, laser_line_y), 15, self.COLOR_SHIELD, 2, 4, 0.5)
                            laser['dodged'] = True
                            # Sound: Shield Block
                    else:
                        if not laser['dodged']:
                            self.player_health -= 10
                            reward -= 2.0
                            self._create_particles( (player_rect.centerx, laser_line_y), 10, self.COLOR_PLAYER_DMG, 1, 3)
                            laser['dodged'] = True
                            # Sound: Player Hit Laser
                elif not laser['dodged'] and laser['pos'].x < player_rect.left:
                    reward += 1.0
                    laser['dodged'] = True

        return reward

    def _create_particles(self, pos, count, color, min_speed, max_speed, life_mult=1.0):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(min_speed, max_speed)
            vel = pygame.Vector2(math.cos(angle) * speed, math.sin(angle) * speed)
            self.particles.append({
                'pos': pygame.Vector2(pos),
                'vel': vel,
                'life': self.np_random.integers(10, 20) * life_mult,
                'color': color,
                'size': self.np_random.integers(2, 4)
            })

    def _get_player_rect(self):
        return pygame.Rect(self.player_pos.x, self.player_pos.y, self.PLAYER_SIZE[0], self.PLAYER_SIZE[1])

    def _get_observation(self):
        render_offset = pygame.Vector2(0, 0)
        if self.screen_shake_timer > 0:
            self.screen_shake_timer -= 1
            render_offset.x = self.np_random.integers(-4, 4)
            render_offset.y = self.np_random.integers(-4, 4)

        # --- Render all game elements ---
        self.screen.fill(self.COLOR_BG)
        self._render_background(render_offset)
        self._render_ground(render_offset)
        self._render_spaceship(render_offset)
        self._render_obstacles(render_offset)
        self._render_lasers(render_offset)
        self._render_player(render_offset)
        self._render_particles(render_offset)
        self._render_ui() # UI not affected by screen shake

        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_background(self, offset):
        for x, y, speed in self.stars:
            star_x = (x - self.world_scroll_x * speed) % self.SCREEN_WIDTH
            pygame.draw.circle(self.screen, self.COLOR_STARS[int(speed > 0.3)], (star_x + offset.x, y + offset.y), 1)

    def _render_ground(self, offset):
        pygame.draw.rect(self.screen, self.COLOR_OBSTACLE, 
                         (offset.x, self.GROUND_Y + offset.y, self.SCREEN_WIDTH, self.SCREEN_HEIGHT - self.GROUND_Y))

    def _render_spaceship(self, offset):
        if self.spaceship_rect.right > 0:
            ship_body = self.spaceship_rect.move(offset.x, offset.y)
            pygame.draw.ellipse(self.screen, self.COLOR_SPACESHIP, ship_body)
            # Cockpit
            cockpit_rect = pygame.Rect(ship_body.centerx - 20, ship_body.top + 20, 40, 40)
            pygame.draw.ellipse(self.screen, self.COLOR_SHIELD, cockpit_rect)
            pygame.gfxdraw.aaellipse(self.screen, cockpit_rect.centerx, cockpit_rect.centery, 20, 20, self.COLOR_SHIELD)

    def _render_obstacles(self, offset):
        for obs in self.obstacles:
            pygame.draw.rect(self.screen, self.COLOR_OBSTACLE, obs['rect'].move(offset.x, offset.y))

    def _render_lasers(self, offset):
        for laser in self.lasers:
            start_pos = laser['pos'] + offset
            if laser['state'] == 'charging':
                alpha = 255 * (math.sin(laser['timer'] * 0.2) * 0.5 + 0.5)
                color = (*self.COLOR_LASER_CHARGE, alpha)
                temp_surf = pygame.Surface((self.SCREEN_WIDTH, self.LASER_HEIGHT), pygame.SRCALPHA)
                pygame.draw.line(temp_surf, color, (0, self.LASER_HEIGHT // 2), (self.SCREEN_WIDTH, self.LASER_HEIGHT // 2), self.LASER_HEIGHT)
                self.screen.blit(temp_surf, (start_pos.x - self.SCREEN_WIDTH, start_pos.y - self.LASER_HEIGHT // 2))
            elif laser['state'] == 'firing':
                pygame.draw.line(self.screen, self.COLOR_LASER_FIRE, (start_pos.x - self.SCREEN_WIDTH, start_pos.y), (start_pos.x, start_pos.y), self.LASER_HEIGHT)
                # Glow effect for laser
                pygame.draw.line(self.screen, (255, 150, 150), (start_pos.x - self.SCREEN_WIDTH, start_pos.y), (start_pos.x, start_pos.y), self.LASER_HEIGHT + 4)

    def _render_player(self, offset):
        player_rect = self._get_player_rect().move(offset.x, offset.y)
        
        # Damage flash
        health_ratio = self.player_health / 100.0
        color = self.COLOR_PLAYER_DMG if self.screen_shake_timer > 0 else self.COLOR_PLAYER
        
        # Player body
        pygame.draw.rect(self.screen, color, player_rect, border_radius=3)

        # Shield
        if self.is_shielding:
            alpha = max(0, min(255, int(150 * (self.shield_timer / self.SHIELD_DURATION))))
            pygame.gfxdraw.aacircle(self.screen, player_rect.centerx, player_rect.centery, self.SHIELD_RADIUS, (*self.COLOR_SHIELD, alpha))
            pygame.gfxdraw.filled_circle(self.screen, player_rect.centerx, player_rect.centery, self.SHIELD_RADIUS, (*self.COLOR_SHIELD, alpha // 4))

    def _render_particles(self, offset):
        for p in self.particles:
            pos = (int(p['pos'].x + offset.x), int(p['pos'].y + offset.y))
            pygame.draw.circle(self.screen, p['color'], pos, int(p['size'] * (p['life'] / 20.0)))
    
    def _render_ui(self):
        # Health Bar
        health_bar_w = 200
        health_bar_h = 20
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR_BG, (10, 10, health_bar_w, health_bar_h))
        current_health_w = max(0, health_bar_w * (self.player_health / 100.0))
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR_FG, (10, 10, current_health_w, health_bar_h))
        
        # Distance to Spaceship
        dist = max(0, self.LEVEL_LENGTH - self.world_scroll_x)
        dist_text = self.font_small.render(f"DISTANCE: {int(dist/100)}m", True, self.COLOR_UI_TEXT)
        self.screen.blit(dist_text, (self.SCREEN_WIDTH - dist_text.get_width() - 10, 10))

        # Score
        score_text = self.font_large.render(f"{int(self.score)}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (self.SCREEN_WIDTH // 2 - score_text.get_width() // 2, self.SCREEN_HEIGHT - 45))
        
        # Shield Cooldown
        if self.shield_cooldown_timer > 0:
            cooldown_ratio = self.shield_cooldown_timer / self.SHIELD_COOLDOWN
            if self.is_shielding:
                cooldown_ratio = self.shield_timer / self.SHIELD_DURATION
            
            bar_width = 50
            bar_height = 8
            y_pos = self.player_pos.y - 15
            x_pos = self.player_pos.x + self.PLAYER_SIZE[0]/2 - bar_width/2
            pygame.draw.rect(self.screen, (50,50,50), (x_pos, y_pos, bar_width, bar_height))
            
            color = self.COLOR_SHIELD if self.is_shielding else (150,150,150)
            pygame.draw.rect(self.screen, color, (x_pos, y_pos, bar_width * cooldown_ratio, bar_height))

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "health": self.player_health,
            "distance_to_goal": max(0, self.LEVEL_LENGTH - self.world_scroll_x)
        }

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        """
        Call this at the end of __init__ to verify implementation.
        """
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
        assert isinstance(trunc, bool)
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game directly
    # To use this, you need to unset the SDL_VIDEODRIVER dummy variable
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]

    env = GameEnv(render_mode="rgb_array")
    # We call reset here because the __init__ no longer does.
    obs, info = env.reset(seed=random.randint(0, 1_000_000))
    
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Alien Escape")
    
    terminated = False
    truncated = False
    total_reward = 0
    
    # Mapping from Pygame keys to MultiDiscrete actions
    key_map = {
        pygame.K_UP: 1, # Unused in current step logic, but kept for potential use
        pygame.K_DOWN: 2, # Unused
        pygame.K_LEFT: 3,
        pygame.K_RIGHT: 4,
    }
    
    while not terminated and not truncated:
        # --- Human Input ---
        movement_action = 0 # No-op
        space_action = 0
        shift_action = 0
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True

        keys = pygame.key.get_pressed()
        # This logic prioritizes left/right over up/down if multiple are pressed
        move_found = False
        for key, move_val in key_map.items():
            if keys[key]:
                movement_action = move_val
                move_found = True
                break 
        
        if keys[pygame.K_SPACE]:
            space_action = 1
            
        if keys[pygame.K_SHIFT]:
            shift_action = 1
        
        action = [movement_action, space_action, shift_action]
        
        # --- Gym Step ---
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # --- Render to Display ---
        # The observation is already a rendered frame, just need to show it
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        if terminated or truncated:
            end_reason = "Terminated" if terminated else "Truncated"
            print(f"Game Over! ({end_reason}) Final Score: {info['score']:.2f}, Steps: {info['steps']}")
            # Wait a bit before closing
            pygame.time.wait(2000)

    env.close()