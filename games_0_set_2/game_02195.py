
# Generated: 2025-08-28T04:01:26.343888
# Source Brief: brief_02195.md
# Brief Index: 2195

        
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
        "Controls: Use ← and → to move. Press ↑ for a small hop, or Space for a large jump. Dodge red obstacles and collect yellow coins!"
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A side-scrolling arcade game where you control a hopping spaceship. Dodge obstacles and collect coins to reach the end of the level."
    )

    # Frames auto-advance at a fixed rate for smooth real-time gameplay.
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Screen and world dimensions
        self.WIDTH, self.HEIGHT = 640, 400
        self.LEVEL_WIDTH = self.WIDTH * 8  # Total length of the level

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
        self.font_ui = pygame.font.SysFont("monospace", 24, bold=True)

        # Colors
        self.COLOR_BG = (15, 15, 25)
        self.COLOR_PLAYER = (60, 160, 255)
        self.COLOR_OBSTACLE = (255, 50, 50)
        self.COLOR_COIN = (255, 220, 0)
        self.COLOR_END = (200, 200, 255)
        self.COLOR_TEXT = (240, 240, 240)

        # Game constants
        self.MAX_STEPS = 2000
        self.GRAVITY = 0.4
        self.GROUND_Y = self.HEIGHT - 50
        self.PLAYER_SPEED_H = 4
        self.JUMP_SMALL = -8
        self.JUMP_LARGE = -12

        # Initialize state variables
        self.player_pos = pygame.Vector2(0, 0)
        self.player_vel = pygame.Vector2(0, 0)
        self.player_size = pygame.Vector2(30, 30)
        self.on_ground = False
        self.world_scroll_x = 0
        self.obstacle_base_speed = 2.5
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.obstacles = []
        self.coins = []
        self.particles = []
        self.stars = []
        self.end_marker = None

        self.reset()
        # self.validate_implementation() # Uncomment for self-testing

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Reset all game state
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.world_scroll_x = 0

        self.player_pos = pygame.Vector2(self.WIDTH / 4, self.GROUND_Y)
        self.player_vel = pygame.Vector2(0, 0)
        self.on_ground = True

        self.obstacles = []
        self.coins = []
        self.particles = []
        
        # Generate level content
        self._generate_level()
        self._generate_stars()
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # --- 1. Unpack Action and Handle Input ---
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        # Horizontal movement
        self.player_vel.x = 0
        if movement == 3: # Left
            self.player_vel.x = -self.PLAYER_SPEED_H
        elif movement == 4: # Right
            self.player_vel.x = self.PLAYER_SPEED_H

        # Jumping
        jump_request = 0
        if movement == 1: # Up for small jump
            jump_request = self.JUMP_SMALL
        if space_held: # Space for large jump (overrides small)
            jump_request = self.JUMP_LARGE

        # --- 2. Update Game Logic & Physics ---
        self.steps += 1
        reward = 0.1  # Base reward for survival

        # Update world scroll and difficulty
        current_speed = self.obstacle_base_speed + (self.steps // 200) * 0.5
        self.world_scroll_x += current_speed

        # Player physics
        if jump_request != 0 and self.on_ground:
            self.player_vel.y = jump_request
            self.on_ground = False
            # sfx: jump sound

        self.player_vel.y += self.GRAVITY
        self.player_pos += self.player_vel

        # Ground collision
        if self.player_pos.y >= self.GROUND_Y:
            self.player_pos.y = self.GROUND_Y
            self.player_vel.y = 0
            if not self.on_ground:
                # sfx: land sound
                self._create_particles(self.player_pos + pygame.Vector2(0, self.player_size.y/2), 5, (200,200,220), 0.5)
            self.on_ground = True
        
        # Screen bounds
        self.player_pos.x = np.clip(self.player_pos.x, 0, self.WIDTH - self.player_size.x)
        self.player_pos.y = np.clip(self.player_pos.y, 0, self.GROUND_Y)

        # Update particles
        for p in self.particles[:]:
            p['pos'] += p['vel']
            p['life'] -= 1
            if p['life'] <= 0:
                self.particles.remove(p)

        # --- 3. Check Interactions & Termination ---
        player_rect = pygame.Rect(self.player_pos.x, self.player_pos.y, self.player_size.x, self.player_size.y)
        
        # Obstacle collisions
        for obs_rect in self.obstacles:
            # Adjust for world scroll
            display_rect = obs_rect.move(-self.world_scroll_x, 0)
            if player_rect.colliderect(display_rect):
                self.game_over = True
                reward = -100
                # sfx: explosion sound
                self._create_particles(self.player_pos + self.player_size/2, 50, self.COLOR_PLAYER, 2)
                break
        
        # Coin collisions
        if not self.game_over:
            for coin in self.coins[:]:
                display_rect = coin['rect'].move(-self.world_scroll_x, 0)
                if player_rect.colliderect(display_rect):
                    self.coins.remove(coin)
                    self.score += 1
                    reward += 5
                    # sfx: coin collect sound
                    self._create_particles(pygame.Vector2(display_rect.center), 20, self.COLOR_COIN, 1)

        # End of level
        end_rect = self.end_marker.move(-self.world_scroll_x, 0)
        if player_rect.colliderect(end_rect):
            self.game_over = True
            reward = 100
            self.score += 10 # Bonus for finishing
            # sfx: level complete sound

        # Max steps termination
        terminated = self.game_over or self.steps >= self.MAX_STEPS

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _get_observation(self):
        # --- Render all game elements ---
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps}

    def _generate_level(self):
        self.obstacles = []
        self.coins = []
        
        current_x = self.WIDTH * 1.5
        while current_x < self.LEVEL_WIDTH - self.WIDTH:
            gap_size = self.np_random.integers(300, 500)
            current_x += gap_size
            
            # Create an obstacle
            obs_h = self.np_random.integers(40, 120)
            obs_w = self.np_random.integers(30, 60)
            self.obstacles.append(pygame.Rect(current_x, self.GROUND_Y - obs_h + self.player_size.y, obs_w, obs_h))

            # Place coins
            coin_pattern = self.np_random.integers(0, 4)
            if coin_pattern == 0: # Arc over obstacle
                for i in range(3):
                    cx = current_x + obs_w/2 - 30 + i * 30
                    cy = self.GROUND_Y - obs_h - 60 + abs(i - 1) * 30
                    self.coins.append({'rect': pygame.Rect(cx, cy, 15, 15), 'angle': i*20})
            elif coin_pattern == 1: # Straight line before
                for i in range(3):
                    self.coins.append({'rect': pygame.Rect(current_x - 100 - i * 40, self.GROUND_Y - 20, 15, 15), 'angle': i*20})
            # other patterns can be added
            
        # End marker
        self.end_marker = pygame.Rect(self.LEVEL_WIDTH, 0, 20, self.HEIGHT)

    def _generate_stars(self):
        self.stars = []
        for _ in range(150):
            self.stars.append({
                'pos': pygame.Vector2(self.np_random.uniform(0, self.LEVEL_WIDTH), self.np_random.uniform(0, self.HEIGHT)),
                'depth': self.np_random.uniform(0.1, 0.8), # For parallax
                'size': self.np_random.uniform(1, 2.5)
            })

    def _create_particles(self, pos, count, color, speed_mult=1.0):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 3) * speed_mult
            self.particles.append({
                'pos': pygame.Vector2(pos),
                'vel': pygame.Vector2(math.cos(angle) * speed, math.sin(angle) * speed),
                'life': self.np_random.integers(20, 40),
                'color': color,
                'size': self.np_random.uniform(2, 5)
            })

    def _render_game(self):
        # Render stars with parallax
        for star in self.stars:
            star_x = (star['pos'].x - self.world_scroll_x * star['depth']) % self.WIDTH
            pygame.draw.circle(self.screen, (255, 255, 255), (int(star_x), int(star['pos'].y)), int(star['size']))

        # Render ground
        pygame.draw.line(self.screen, self.COLOR_END, (0, int(self.GROUND_Y + self.player_size.y/2)), (self.WIDTH, int(self.GROUND_Y + self.player_size.y/2)), 3)

        # Render end marker
        end_rect = self.end_marker.move(-self.world_scroll_x, 0)
        if end_rect.right > 0:
            pygame.draw.rect(self.screen, self.COLOR_END, end_rect)

        # Render obstacles
        for obs_rect in self.obstacles:
            display_rect = obs_rect.move(-self.world_scroll_x, 0)
            if -obs_rect.width < display_rect.x < self.WIDTH:
                pygame.gfxdraw.box(self.screen, display_rect, self.COLOR_OBSTACLE)
                pygame.gfxdraw.rectangle(self.screen, display_rect, (255, 150, 150))

        # Render coins
        for coin in self.coins:
            display_rect = coin['rect'].move(-self.world_scroll_x, 0)
            if -coin['rect'].width < display_rect.x < self.WIDTH:
                coin['angle'] = (coin['angle'] + 10) % 360
                w = coin['rect'].width
                h = coin['rect'].height * abs(math.cos(math.radians(coin['angle'])))
                
                if h > 1:
                    r = pygame.Rect(display_rect.centerx - w/2, display_rect.centery - h/2, w, h)
                    pygame.gfxdraw.ellipse(self.screen, int(r.centerx), int(r.centery), int(w/2), int(h/2), self.COLOR_COIN)
                    pygame.gfxdraw.ellipse(self.screen, int(r.centerx), int(r.centery), int(w/2-1), int(h/2-1), (255, 255, 150))

        # Render particles
        for p in self.particles:
            alpha = int(255 * (p['life'] / 40))
            color = (*p['color'], alpha)
            pygame.gfxdraw.filled_circle(self.screen, int(p['pos'].x), int(p['pos'].y), int(p['size'] * (p['life'] / 40)), color)

        # Render player
        if not self.game_over:
            # Squash and stretch effect
            squash = 1.0 - min(0.4, max(-0.4, self.player_vel.y * 0.05))
            stretch = 1.0 + min(0.5, max(-0.3, self.player_vel.y * -0.05))
            w, h = self.player_size.x * squash, self.player_size.y * stretch
            
            player_rect = pygame.Rect(self.player_pos.x, self.player_pos.y + (self.player_size.y - h), w, h)
            
            # Simple polygon for spaceship
            p1 = (player_rect.centerx, player_rect.top)
            p2 = (player_rect.left, player_rect.bottom)
            p3 = (player_rect.right, player_rect.bottom)
            
            pygame.gfxdraw.aapolygon(self.screen, [p1, p2, p3], self.COLOR_PLAYER)
            pygame.gfxdraw.filled_polygon(self.screen, [p1, p2, p3], self.COLOR_PLAYER)
            
            # Engine trail
            if not self.on_ground:
                self._create_particles(self.player_pos + pygame.Vector2(self.player_size.x/2, self.player_size.y), 1, (200, 200, 255), 0.2)

    def _render_ui(self):
        score_text = self.font_ui.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))

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
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game directly
    # Set `human_render` to True to see the Pygame window
    human_render = True
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()

    if human_render:
        pygame.display.set_caption("Arcade Hopper")
        game_screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))

    terminated = False
    total_reward = 0
    
    # --- Action mapping for human play ---
    # action = [movement, space, shift]
    # movement: 0=none, 1=up, 2=down, 3=left, 4=right
    action = [0, 0, 0]

    while not terminated:
        if human_render:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    terminated = True
            
            keys = pygame.key.get_pressed()
            action = [0, 0, 0] # Reset action each frame
            
            # Movement
            if keys[pygame.K_LEFT]:
                action[0] = 3
            elif keys[pygame.K_RIGHT]:
                action[0] = 4
            elif keys[pygame.K_UP]:
                action[0] = 1 # Small jump
            
            # Buttons
            if keys[pygame.K_SPACE]:
                action[1] = 1 # Large jump
            if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
                action[2] = 1

        else: # For testing without human input
            action = env.action_space.sample()

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if human_render:
            # The observation is already a rendered frame
            # We just need to convert it back to a Pygame surface to display
            surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
            game_screen.blit(surf, (0, 0))
            pygame.display.flip()
            env.clock.tick(60) # Limit to 60 FPS for human play

        if terminated:
            print(f"Episode finished. Score: {info['score']}, Total Reward: {total_reward:.2f}, Steps: {info['steps']}")
            obs, info = env.reset()
            total_reward = 0
            if not human_render: # Stop after one episode if not in human mode
                break

    env.close()