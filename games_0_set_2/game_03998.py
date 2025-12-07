import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import os
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Use ← and → to run, and ↑ to jump. Collect all the coins and reach the green flag!"
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A fast-paced, procedurally generated platformer. Guide your robot to the finish line, collecting coins and avoiding falls."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Gymnasium spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen_width = 640
        self.screen_height = 400
        self.screen = pygame.Surface((self.screen_width, self.screen_height))
        self.clock = pygame.time.Clock()
        
        # Fonts
        self.font_ui = pygame.font.SysFont("monospace", 20, bold=True)
        self.font_game_over = pygame.font.SysFont("monospace", 48, bold=True)

        # Colors
        self.COLOR_BG = (15, 15, 25)
        self.COLOR_GRID = (30, 30, 45)
        self.COLOR_PLAYER = (60, 160, 255)
        self.COLOR_PLAYER_HEAD = (120, 200, 255)
        self.COLOR_PLATFORM = (100, 100, 120)
        self.COLOR_PLATFORM_LIT = (180, 180, 200)
        self.COLOR_COIN = (255, 220, 0)
        self.COLOR_FINISH = (0, 255, 100)
        self.COLOR_TEXT = (240, 240, 240)
        self.COLOR_PARTICLE_JUMP = (200, 200, 255)
        self.COLOR_PARTICLE_COIN = (255, 240, 150)

        # Game constants
        self.GRAVITY = 0.4
        self.FRICTION = -0.12
        self.PLAYER_ACCEL = 0.6
        self.PLAYER_MAX_SPEED = 5
        self.JUMP_STRENGTH = -9.5
        self.MAX_STEPS = 1500

        # Initialize state variables
        self.player_pos = None
        self.player_vel = None
        self.player_size = None
        self.on_ground = None
        self.platforms = None
        self.coins = None
        self.finish_line = None
        self.particles = None
        self.camera_pos = None
        self.steps = None
        self.score = None
        self.game_over = None
        self.win = None
        
        # self.reset() is called by the wrapper, but for standalone use it's good practice
        # However, to avoid issues with gym.make, we let the first reset be called externally.

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initialize all game state
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False
        
        self.player_pos = np.array([100.0, 300.0])
        self.player_vel = np.array([0.0, 0.0])
        self.player_size = np.array([24, 32])
        self.on_ground = False
        
        self.particles = []
        self._generate_level()

        self.camera_pos = np.array([0.0, 0.0])
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # Unpack factorized action
        movement = action[0]  # 0-4: none/up/down/left/right
        
        reward = 0
        
        # --- Game Logic ---
        self._handle_input(movement)
        self._update_player_physics()
        
        # Store position before collision for reward calculation
        prev_x = self.player_pos[0]
        
        self._handle_collisions()
        
        # Reward for horizontal progress
        if self.player_pos[0] > prev_x:
            reward += 0.1
        
        collected_coins = self._handle_coin_collection()
        reward += collected_coins * 1.0
        self.score += collected_coins

        # --- Termination ---
        terminated = False
        self._check_termination_conditions()
        if self.game_over:
            terminated = True
            if self.win:
                reward += 100.0
            else: # Fell
                reward -= 100.0

        truncated = self.steps >= self.MAX_STEPS
        if truncated:
            terminated = True # Gymnasium standard: terminated is True if truncated is True

        self.steps += 1
        
        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _handle_input(self, movement):
        # Horizontal movement
        if movement == 3:  # Left
            self.player_vel[0] -= self.PLAYER_ACCEL
        elif movement == 4:  # Right
            self.player_vel[0] += self.PLAYER_ACCEL
        
        # Jumping
        if movement == 1 and self.on_ground:  # Up
            self.player_vel[1] = self.JUMP_STRENGTH
            self.on_ground = False
            # Spawn jump particles
            for _ in range(10):
                self.particles.append(self._create_particle(
                    self.player_pos + [self.player_size[0] / 2, self.player_size[1]],
                    self.COLOR_PARTICLE_JUMP,
                    life=20,
                    angle_range=(225, 315),
                    speed_range=(1, 4)
                ))

    def _update_player_physics(self):
        # Apply friction
        if abs(self.player_vel[0]) > 0.1:
            self.player_vel[0] += self.FRICTION * np.sign(self.player_vel[0])
        else:
            self.player_vel[0] = 0

        # Clamp horizontal speed
        self.player_vel[0] = np.clip(self.player_vel[0], -self.PLAYER_MAX_SPEED, self.PLAYER_MAX_SPEED)
        
        # Apply gravity
        self.player_vel[1] += self.GRAVITY
        self.player_vel[1] = min(self.player_vel[1], 10) # Terminal velocity

        # Update position
        self.player_pos += self.player_vel

    def _handle_collisions(self):
        self.on_ground = False
        player_rect = pygame.Rect(self.player_pos, self.player_size)

        for platform_data in self.platforms:
            platform_rect = platform_data['rect']
            if player_rect.colliderect(platform_rect):
                # Check if player was above the platform in the previous frame
                prev_player_bottom = self.player_pos[1] - self.player_vel[1] + self.player_size[1]

                if self.player_vel[1] >= 0 and prev_player_bottom <= platform_rect.top + 1:
                    # Land on top
                    self.player_pos[1] = platform_rect.top - self.player_size[1]
                    self.player_vel[1] = 0
                    self.on_ground = True
                    platform_data['lit_timer'] = 10 # Cause platform to flash
                elif self.player_vel[1] < 0 and player_rect.top < platform_rect.bottom:
                    # Bonk head on bottom
                    self.player_pos[1] = platform_rect.bottom
                    self.player_vel[1] = 0
                else:
                    # Side collision
                    if self.player_vel[0] > 0: # Moving right
                        self.player_pos[0] = platform_rect.left - self.player_size[0]
                    elif self.player_vel[0] < 0: # Moving left
                        self.player_pos[0] = platform_rect.right
                    self.player_vel[0] = 0

    def _handle_coin_collection(self):
        player_rect = pygame.Rect(self.player_pos, self.player_size)
        collected_count = 0
        
        remaining_coins = []
        for coin in self.coins:
            coin_rect = pygame.Rect(coin['pos'][0] - 8, coin['pos'][1] - 8, 16, 16)
            if player_rect.colliderect(coin_rect):
                collected_count += 1
                # Spawn coin collection particles
                for _ in range(15):
                    self.particles.append(self._create_particle(
                        coin['pos'],
                        self.COLOR_PARTICLE_COIN,
                        life=25,
                        angle_range=(0, 360),
                        speed_range=(0.5, 3)
                    ))
            else:
                remaining_coins.append(coin)
        
        self.coins = remaining_coins
        return collected_count

    def _check_termination_conditions(self):
        # Fall off screen
        if self.player_pos[1] > self.screen_height:
            self.game_over = True
            self.win = False

        # Reach finish line
        player_rect = pygame.Rect(self.player_pos, self.player_size)
        if self.finish_line and player_rect.colliderect(self.finish_line):
            self.game_over = True
            self.win = True

    def _generate_level(self):
        self.platforms = []
        self.coins = []

        # Start platform
        start_rect = pygame.Rect(0, 350, 300, 50)
        self.platforms.append({'rect': start_rect, 'lit_timer': 0})

        # Procedural generation
        x = start_rect.right
        y = start_rect.y
        for i in range(20):
            gap = self.np_random.uniform(40, 100)
            x += gap
            
            y_change = self.np_random.uniform(-80, 80)
            y += y_change
            y = np.clip(y, 200, 350)
            
            width = self.np_random.uniform(80, 250)
            
            new_rect = pygame.Rect(x, y, width, 20)
            self.platforms.append({'rect': new_rect, 'lit_timer': 0})
            
            # Add coins above the platform
            num_coins = self.np_random.integers(0, 4)
            for j in range(num_coins):
                coin_x = x + (j + 1) * (width / (num_coins + 1))
                coin_y = y - 40 - self.np_random.uniform(0, 30)
                self.coins.append({
                    'pos': np.array([coin_x, coin_y]),
                    'anim_offset': self.np_random.uniform(0, 2 * math.pi)
                })

            x += width

        # Finish line
        last_platform_rect = self.platforms[-1]['rect']
        self.finish_line = pygame.Rect(last_platform_rect.centerx - 10, last_platform_rect.y - 60, 20, 60)

    def _update_camera(self):
        # Smooth horizontal follow
        target_x = self.player_pos[0] - self.screen_width / 2.5
        self.camera_pos[0] += (target_x - self.camera_pos[0]) * 0.1

        # Smooth vertical follow, with a delay
        target_y = self.player_pos[1] - self.screen_height / 1.8
        self.camera_pos[1] += (target_y - self.camera_pos[1]) * 0.05
        
        # Clamp camera
        self.camera_pos[0] = max(0, self.camera_pos[0])
        self.camera_pos[1] = min(100, self.camera_pos[1])

    def _get_observation(self):
        self._update_camera()

        # --- Rendering ---
        self.screen.fill(self.COLOR_BG)
        self._render_background()
        self._render_platforms()
        self._render_coins()
        self._render_finish_line()
        self._render_particles()
        self._render_player()
        self._render_ui()
        
        if self.game_over:
            self._render_game_over()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_background(self):
        # Draw grid
        cam_x_offset = self.camera_pos[0] % 40
        cam_y_offset = self.camera_pos[1] % 40
        for x in range(-40, self.screen_width + 40, 40):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x - cam_x_offset, 0), (x - cam_x_offset, self.screen_height))
        for y in range(-40, self.screen_height + 40, 40):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y - cam_y_offset), (self.screen_width, y - cam_y_offset))
            
    def _render_platforms(self):
        for p_data in self.platforms:
            p_rect = p_data['rect']
            draw_pos = (int(p_rect.x - self.camera_pos[0]), int(p_rect.y - self.camera_pos[1]))
            color = self.COLOR_PLATFORM
            if p_data['lit_timer'] > 0:
                color = self.COLOR_PLATFORM_LIT
                p_data['lit_timer'] -= 1
            pygame.draw.rect(self.screen, color, (draw_pos, p_rect.size))

    def _render_coins(self):
        for coin in self.coins:
            t = self.steps * 0.1 + coin['anim_offset']
            y_offset = math.sin(t * 1.5) * 4
            width_scale = (math.sin(t) * 0.4 + 0.6)
            
            pos = coin['pos'] - self.camera_pos
            x, y = int(pos[0]), int(pos[1] + y_offset)
            
            width = int(16 * width_scale)
            height = 16
            
            if width > 1:
                pygame.gfxdraw.ellipse(self.screen, x, y, width // 2, height // 2, self.COLOR_COIN)
                pygame.gfxdraw.filled_ellipse(self.screen, x, y, width // 2, height // 2, self.COLOR_COIN)

    def _render_finish_line(self):
        if not self.finish_line: return
        # Pole
        pole_rect = pygame.Rect(
            self.finish_line.x - self.camera_pos[0],
            self.finish_line.y - self.camera_pos[1],
            4, self.finish_line.height
        )
        pygame.draw.rect(self.screen, self.COLOR_PLATFORM_LIT, pole_rect)
        
        # Flag
        flag_points = [
            (pole_rect.right, pole_rect.top),
            (pole_rect.right + 30 + math.sin(self.steps * 0.1) * 3, pole_rect.top + 10),
            (pole_rect.right, pole_rect.top + 20)
        ]
        pygame.gfxdraw.aapolygon(self.screen, flag_points, self.COLOR_FINISH)
        pygame.gfxdraw.filled_polygon(self.screen, flag_points, self.COLOR_FINISH)

    def _render_player(self):
        # Squash and stretch effect
        squash = max(0, -self.player_vel[1] * 0.5) if not self.on_ground else min(10, abs(self.player_vel[1]) * 2)
        stretch = max(0, self.player_vel[1] * 0.5)
        
        body_w = self.player_size[0] + squash - stretch
        body_h = self.player_size[1] - squash + stretch
        
        # Tilt effect
        tilt = self.player_vel[0] * 2.5
        
        # Position
        draw_pos = self.player_pos - self.camera_pos
        
        # Body
        body_rect = pygame.Rect(
            draw_pos[0],
            draw_pos[1] + (self.player_size[1] - body_h),
            body_w,
            body_h
        )
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, body_rect, border_radius=4)
        
        # Head
        head_center = (
            int(body_rect.centerx + tilt),
            int(body_rect.top - 8)
        )
        pygame.gfxdraw.aacircle(self.screen, head_center[0], head_center[1], 8, self.COLOR_PLAYER_HEAD)
        pygame.gfxdraw.filled_circle(self.screen, head_center[0], head_center[1], 8, self.COLOR_PLAYER_HEAD)

    def _render_particles(self):
        for p in self.particles[:]:
            p['pos'] += p['vel']
            p['life'] -= 1
            if p['life'] <= 0:
                self.particles.remove(p)
            else:
                draw_pos = p['pos'] - self.camera_pos
                size = int(max(0, (p['life'] / p['start_life']) * 5))
                if size > 0:
                    pygame.draw.circle(self.screen, p['color'], (int(draw_pos[0]), int(draw_pos[1])), size)

    def _render_ui(self):
        score_text = self.font_ui.render(f"COINS: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))

    def _render_game_over(self):
        s = pygame.Surface((self.screen_width, self.screen_height), pygame.SRCALPHA)
        s.fill((0, 0, 0, 180))
        
        message = "LEVEL COMPLETE!" if self.win else "GAME OVER"
        color = self.COLOR_FINISH if self.win else (255, 80, 80)
        
        text = self.font_game_over.render(message, True, color)
        text_rect = text.get_rect(center=(self.screen_width / 2, self.screen_height / 2))
        s.blit(text, text_rect)
        self.screen.blit(s, (0, 0))

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "player_pos": self.player_pos.tolist(),
            "win": self.win
        }

    def _create_particle(self, pos, color, life, angle_range, speed_range):
        angle = math.radians(self.np_random.uniform(*angle_range))
        speed = self.np_random.uniform(*speed_range)
        vel = np.array([math.cos(angle), math.sin(angle)]) * speed
        return {
            'pos': pos.copy(),
            'vel': vel,
            'life': life,
            'start_life': life,
            'color': color
        }

    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # This block allows you to play the game directly
    # In this mode, we need a display
    os.environ["SDL_VIDEODRIVER"] = "x11"
    
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((640, 400))
    pygame.display.set_caption("Robot Platformer")
    
    running = True
    total_reward = 0
    
    # Game loop
    while running:
        movement = 0 # No-op
        keys = pygame.key.get_pressed()
        
        if keys[pygame.K_LEFT]:
            movement = 3
        elif keys[pygame.K_RIGHT]:
            movement = 4
        
        if keys[pygame.K_UP]:
            movement = 1
        
        # The action space is MultiDiscrete, but for human play we only need movement
        action = [movement, 0, 0] 
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # Render the observation from the environment to the screen
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
            # Wait a bit before resetting
            pygame.time.wait(2000)
            obs, info = env.reset()
            total_reward = 0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        env.clock.tick(60) # Control the frame rate

    env.close()