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
        "Controls: ←→ to run, ↑ or Space to jump. Collect coins and reach the green flag!"
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A fast-paced pixel art platformer. Jump across platforms, collect coins, and reach the flag before time runs out. Avoid falling into pits!"
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    LEVEL_WIDTH = 3200
    FPS = 30
    TIME_LIMIT_SECONDS = 60

    # Colors
    COLOR_BG = (135, 206, 235)  # Sky Blue
    COLOR_PLAYER = (255, 50, 50) # Bright Red
    COLOR_PLATFORM = (139, 69, 19) # Brown
    COLOR_PLATFORM_TOP = (160, 82, 45) # Lighter Brown
    COLOR_COIN = (255, 215, 0) # Gold
    COLOR_PIT = (20, 20, 20) # Near Black
    COLOR_FLAG_POLE = (192, 192, 192) # Silver
    COLOR_FLAG = (0, 200, 0) # Green
    COLOR_TEXT = (255, 255, 255)
    COLOR_TEXT_SHADOW = (50, 50, 50)
    
    # Physics
    GRAVITY = 0.7
    PLAYER_ACCEL = 1.0
    PLAYER_FRICTION = 0.85
    PLAYER_JUMP_STRENGTH = -14
    MAX_PLAYER_SPEED = 7

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
        self.player_pos = None
        self.player_vel = None
        self.on_ground = False
        self.player_facing_right = True
        self.jump_request = False
        
        self.camera_x = 0
        self.particles = []
        self.coins = []
        self.platforms = []
        self.pits = []
        self.flag_rect = None

        self.steps = 0
        self.score = 0
        self.time_left = 0
        
        # The validation function is called to ensure the environment conforms
        # to the Gym API and our specific requirements.
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.time_left = self.FPS * self.TIME_LIMIT_SECONDS
        
        # Player state
        self.player_pos = np.array([100.0, 200.0])
        self.player_vel = np.array([0.0, 0.0])
        self.on_ground = False
        self.player_facing_right = True
        self.jump_request = False

        # Level Generation
        self._generate_level()

        self.particles = []
        
        return self._get_observation(), self._get_info()
    
    def _generate_level(self):
        self.platforms = [
            pygame.Rect(0, 350, 400, 50),
            pygame.Rect(500, 300, 200, 50),
            pygame.Rect(800, 250, 250, 50),
            pygame.Rect(1150, 300, 150, 50),
            pygame.Rect(1400, 220, 100, 20),
            pygame.Rect(1600, 280, 100, 20),
            pygame.Rect(1800, 240, 100, 20),
            pygame.Rect(2100, 300, 400, 100),
            pygame.Rect(2600, 250, 150, 20),
            pygame.Rect(2850, 200, 150, 20),
            pygame.Rect(2950, 350, 250, 50),
        ]
        self.pits = [
            pygame.Rect(400, 380, 100, 20),
            pygame.Rect(1300, 380, 100, 20),
            pygame.Rect(1500, 380, 600, 20),
            pygame.Rect(2500, 380, 100, 20),
            pygame.Rect(2750, 380, 100, 20),
        ]
        self.coins = [
            {'pos': np.array([p.centerx, p.y - 40]), 'collected': False, 'anim_offset': self.np_random.uniform(0, math.pi)}
            for p in [self.platforms[1], self.platforms[2], self.platforms[4], self.platforms[6], self.platforms[8], self.platforms[9]]
        ]
        self.flag_rect = pygame.Rect(3050, 250, 20, 100)


    def step(self, action):
        # Unpack factorized action
        movement = action[0]  # 0-4: none/up/down/left/right
        space_held = action[1] == 1
        
        reward = 0.01  # Survival reward
        terminated = False
        
        # --- Handle Input ---
        target_vx = 0
        if movement == 3:  # Left
            target_vx = -self.MAX_PLAYER_SPEED
            self.player_facing_right = False
        elif movement == 4: # Right
            target_vx = self.MAX_PLAYER_SPEED
            self.player_facing_right = True

        if self.on_ground and (movement == 1 or space_held):
            self.player_vel[1] = self.PLAYER_JUMP_STRENGTH
            self.on_ground = False
            self.jump_request = False
            # sfx: jump
            self._create_particles(self.player_pos + np.array([15, 30]), (200, 200, 200), 10)

        # --- Update Physics ---
        # Horizontal movement with acceleration
        self.player_vel[0] += (target_vx - self.player_vel[0]) * (1 - self.PLAYER_FRICTION)

        # Vertical movement (gravity)
        self.player_vel[1] += self.GRAVITY
        
        # --- Collision Detection & Resolution ---
        player_rect = pygame.Rect(self.player_pos[0], self.player_pos[1], 30, 30)
        
        # Horizontal collision
        self.player_pos[0] += self.player_vel[0]
        player_rect.x = int(self.player_pos[0])
        for plat in self.platforms:
            if player_rect.colliderect(plat):
                if self.player_vel[0] > 0: # Moving right
                    player_rect.right = plat.left
                elif self.player_vel[0] < 0: # Moving left
                    player_rect.left = plat.right
                self.player_pos[0] = player_rect.x
                self.player_vel[0] = 0
        
        # Vertical collision
        self.on_ground = False
        self.player_pos[1] += self.player_vel[1]
        player_rect.y = int(self.player_pos[1])
        for plat in self.platforms:
            if player_rect.colliderect(plat):
                if self.player_vel[1] > 0: # Falling
                    player_rect.bottom = plat.top
                    self.on_ground = True
                    self.player_vel[1] = 0
                elif self.player_vel[1] < 0: # Jumping
                    player_rect.top = plat.bottom
                    self.player_vel[1] = 0 # Bonk head
                self.player_pos[1] = player_rect.y

        # Keep player within level bounds
        self.player_pos[0] = np.clip(self.player_pos[0], 0, self.LEVEL_WIDTH - player_rect.width)

        # --- Other Interactions ---
        player_rect = pygame.Rect(self.player_pos[0], self.player_pos[1], 30, 30)

        # Coins
        for coin in self.coins:
            if not coin['collected']:
                coin_rect = pygame.Rect(coin['pos'][0] - 10, coin['pos'][1] - 10, 20, 20)
                if player_rect.colliderect(coin_rect):
                    coin['collected'] = True
                    self.score += 1
                    reward += 10
                    # sfx: coin_collect
                    self._create_particles(coin['pos'], self.COLOR_COIN, 15, 0.5)
        
        # Pits
        for pit in self.pits:
            if player_rect.colliderect(pit):
                terminated = True
                reward = -100
                # sfx: fall_scream

        # Out of bounds (fall)
        if self.player_pos[1] > self.SCREEN_HEIGHT + 50:
            terminated = True
            reward = -100
            # sfx: fall_scream

        # Flag
        if player_rect.colliderect(self.flag_rect):
            terminated = True
            reward += 100
            # sfx: victory_fanfare

        # --- Update Game State ---
        self.steps += 1
        self.time_left -= 1
        if self.time_left <= 0:
            terminated = True
            
        self._update_particles()
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )
    
    def _get_observation(self):
        # Update camera to follow player
        self.camera_x = self.player_pos[0] - self.SCREEN_WIDTH / 2
        self.camera_x = np.clip(self.camera_x, 0, self.LEVEL_WIDTH - self.SCREEN_WIDTH)
        
        # --- Rendering ---
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Parallax background mountains
        for i in range(5):
            x = (100 + i * 800 - self.camera_x * 0.2) % (self.SCREEN_WIDTH + 800) - 400
            pygame.gfxdraw.filled_trigon(self.screen, int(x), 350, int(x+200), 100, int(x+400), 350, (100, 150, 180))
        
        # Platforms
        for plat in self.platforms:
            draw_rect = plat.move(-self.camera_x, 0)
            pygame.draw.rect(self.screen, self.COLOR_PLATFORM, draw_rect)
            pygame.draw.rect(self.screen, self.COLOR_PLATFORM_TOP, (draw_rect.x, draw_rect.y, draw_rect.width, 5))
            
        # Pits
        for pit in self.pits:
            pygame.draw.rect(self.screen, self.COLOR_PIT, pit.move(-self.camera_x, 0))

        # Coins
        for coin in self.coins:
            if not coin['collected']:
                anim_width = 10 * abs(math.sin(self.steps * 0.2 + coin['anim_offset']))
                pos_x = int(coin['pos'][0] - self.camera_x)
                pos_y = int(coin['pos'][1])
                coin_rect = pygame.Rect(pos_x - anim_width, pos_y - 10, anim_width * 2, 20)
                pygame.draw.ellipse(self.screen, self.COLOR_COIN, coin_rect)
                pygame.draw.ellipse(self.screen, (255, 255, 100), coin_rect, 2)
        
        # Flag
        pole_rect = self.flag_rect.move(-self.camera_x, 0)
        flag_points = [
            (pole_rect.right, pole_rect.top),
            (pole_rect.right + 50, pole_rect.top + 25),
            (pole_rect.right, pole_rect.top + 50)
        ]
        pygame.draw.rect(self.screen, self.COLOR_FLAG_POLE, pole_rect)
        pygame.draw.polygon(self.screen, self.COLOR_FLAG, flag_points)

        # Particles
        for p in self.particles:
            p_x = int(p['pos'][0] - self.camera_x)
            p_y = int(p['pos'][1])
            alpha_color = p['color'] + (p['alpha'],)
            temp_surf = pygame.Surface((p['size']*2, p['size']*2), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, alpha_color, (p['size'], p['size']), p['size'])
            self.screen.blit(temp_surf, (p_x - p['size'], p_y - p['size']))

        # Player
        player_draw_pos = (int(self.player_pos[0] - self.camera_x), int(self.player_pos[1]))
        player_rect = pygame.Rect(player_draw_pos[0], player_draw_pos[1], 30, 30)
        pygame.draw.rect(self.screen, (0,0,0), player_rect.inflate(4,4))
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, player_rect)
        
        # Player eye
        eye_x = player_rect.centerx + (8 if self.player_facing_right else -8)
        eye_y = player_rect.centery - 5
        pygame.draw.circle(self.screen, (255,255,255), (eye_x, eye_y), 4)
        pygame.draw.circle(self.screen, (0,0,0), (eye_x, eye_y), 2)


    def _render_ui(self):
        # Helper to draw text with shadow
        def draw_text_shadow(text, font, color, pos):
            img_shadow = font.render(text, True, self.COLOR_TEXT_SHADOW)
            self.screen.blit(img_shadow, (pos[0] + 2, pos[1] + 2))
            img = font.render(text, True, color)
            self.screen.blit(img, pos)

        # Score
        score_text = f"COINS: {self.score}"
        draw_text_shadow(score_text, self.font_small, self.COLOR_TEXT, (10, 10))

        # Time
        time_str = f"TIME: {self.time_left // self.FPS:02d}"
        time_color = self.COLOR_TEXT if self.time_left // self.FPS > 10 else (255, 100, 100)
        time_size = self.font_small.size(time_str)
        draw_text_shadow(time_str, self.font_small, time_color, (self.SCREEN_WIDTH - time_size[0] - 10, 10))

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "time_left": self.time_left,
        }

    def _create_particles(self, pos, color, count, speed_mult=1.0):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4) * speed_mult
            vel = np.array([math.cos(angle) * speed, math.sin(angle) * speed])
            self.particles.append({
                'pos': pos.copy(),
                'vel': vel,
                'color': color,
                'alpha': 255,
                'size': self.np_random.integers(3, 7)
            })

    def _update_particles(self):
        for p in self.particles:
            p['pos'] += p['vel']
            p['alpha'] -= 10
            p['size'] = max(0, p['size'] - 0.1)
        self.particles = [p for p in self.particles if p['alpha'] > 0]

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation:
        '''
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test reset (this also initializes the state for other tests)
        obs, info = self.reset()
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(info, dict)
        
        # Test observation space (now that state is initialized)
        test_obs = self._get_observation()
        assert test_obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == "__main__":
    # This block allows you to play the game directly
    # It will not run in a headless environment.
    # To enable rendering, comment out the `os.environ` line at the top.
    try:
        del os.environ["SDL_VIDEODRIVER"]
    except KeyError:
        pass

    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # Setup Pygame window for human play
    pygame.display.set_caption("Pixel Platformer")
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    
    terminated = False
    running = True
    total_reward = 0

    while running:
        # --- Human Input ---
        movement = 0 # No-op
        space = 0
        shift = 0 # Unused in this game
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            movement = 3
        if keys[pygame.K_RIGHT]:
            movement = 4
        if keys[pygame.K_UP]:
            movement = 1
        if keys[pygame.K_SPACE]:
            space = 1
            
        action = [movement, space, shift]
        
        # --- Gym Step ---
        if not terminated:
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward

        # --- Pygame Event Handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r and terminated:
                    print("--- RESETTING ---")
                    obs, info = env.reset()
                    terminated = False
                    total_reward = 0
                if event.key == pygame.K_ESCAPE:
                    running = False

        # --- Rendering ---
        # The observation is already a rendered frame
        # We just need to display it
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))

        if terminated:
            # Display game over message
            font = pygame.font.Font(None, 64)
            text = "GAME OVER"
            # Example win condition based on reaching the flag (high reward)
            if reward > 50:
                text = "YOU WIN!"
            
            text_surf = font.render(text, True, (255, 255, 255))
            text_rect = text_surf.get_rect(center=(GameEnv.SCREEN_WIDTH / 2, GameEnv.SCREEN_HEIGHT / 2 - 30))
            shadow_surf = font.render(text, True, (50, 50, 50))
            screen.blit(shadow_surf, text_rect.move(3,3))
            screen.blit(text_surf, text_rect)
            
            font_small = pygame.font.Font(None, 32)
            reset_text = "Press 'R' to Restart"
            reset_surf = font_small.render(reset_text, True, (255, 255, 255))
            reset_rect = reset_surf.get_rect(center=(GameEnv.SCREEN_WIDTH / 2, GameEnv.SCREEN_HEIGHT / 2 + 30))
            screen.blit(reset_surf, reset_rect)


        pygame.display.flip()
        env.clock.tick(GameEnv.FPS)

    env.close()