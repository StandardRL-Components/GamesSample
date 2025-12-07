import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
from collections import deque
import os
import os
import pygame


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: ←→ to run, ↑ to jump. Collect coins and reach the green flag before time runs out!"
    )

    game_description = (
        "A fast-paced, procedurally generated platformer. Jump across gaps, ride moving platforms, "
        "and collect coins to maximize your score. Reach the flag to win!"
    )

    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Game Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 30
        self.MAX_STEPS = 1000
        self.TIME_LIMIT_SECONDS = 60

        # --- Pygame Setup (Headless) ---
        os.environ['SDL_VIDEODRIVER'] = 'dummy'
        pygame.init()
        pygame.font.init()

        # --- Colors ---
        self.COLOR_BG = (135, 206, 235)  # Sky Blue
        self.COLOR_PLAYER = (255, 87, 34) # Deep Orange
        self.COLOR_PLAYER_GLOW = (255, 138, 101) # Lighter Orange
        self.COLOR_STATIC_PLATFORM = (139, 69, 19) # Saddle Brown
        self.COLOR_MOVING_PLATFORM = (220, 20, 60) # Crimson
        self.COLOR_COLLAPSING_PLATFORM = (112, 128, 144) # Slate Gray
        self.COLOR_COIN = (255, 215, 0) # Gold
        self.COLOR_FLAG = (34, 139, 34) # Forest Green
        self.COLOR_TEXT = (255, 255, 255)
        self.COLOR_TEXT_SHADOW = (0, 0, 0, 150)
        self.COLOR_PARTICLE = (240, 230, 140) # Khaki

        # --- Physics & Player ---
        self.GRAVITY = 0.8
        self.JUMP_FORCE = -15
        self.PLAYER_SPEED = 6
        self.PLAYER_SIZE = 24
        self.TERMINAL_VELOCITY = 18

        # --- Gymnasium Spaces ---
        self.observation_space = Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Surfaces & Fonts ---
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.Font(None, 48)
        self.font_small = pygame.font.Font(None, 32)
        
        # --- Game State ---
        self.player_rect = None
        self.player_vel = None
        self.on_ground = None
        self.camera_x = None
        self.platforms = None
        self.coins = None
        self.flag_rect = None
        self.particles = None
        
        self.steps = 0
        self.score = 0
        self.time_left = 0
        self.level_end_x = 0
        self.max_dist_reached = 0
        
        # Difficulty scaling parameters
        self.gap_increase = 0
        self.speed_increase = 0.0

        # Initialize state variables
        # A seed is not passed here, but the super().reset() in the reset method will handle it.
        # self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # --- Reset Game State ---
        self.steps = 0
        self.score = 0
        self.time_left = self.TIME_LIMIT_SECONDS * self.FPS
        
        player_start_pos = (100, self.HEIGHT - 100)
        self.player_rect = pygame.Rect(player_start_pos[0], player_start_pos[1], self.PLAYER_SIZE, self.PLAYER_SIZE)
        self.player_vel = pygame.Vector2(0, 0)
        self.on_ground = False

        self.camera_x = 0
        self.max_dist_reached = self.player_rect.x
        
        self.particles = deque(maxlen=100) # Efficient particle management

        # Difficulty scaling
        self.gap_increase = (self.np_random.integers(0, 500) // 500) * 2
        self.speed_increase = (self.np_random.integers(0, 500) // 500) * 0.05
        
        self._generate_level()

        return self._get_observation(), self._get_info()

    def step(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        reward = -0.02  # Time penalty

        # --- 1. Update Player Horizontal Movement ---
        if movement == 3:  # Left
            self.player_vel.x = -self.PLAYER_SPEED
        elif movement == 4:  # Right
            self.player_vel.x = self.PLAYER_SPEED
        else:
            self.player_vel.x = 0

        # --- 2. Update Player Vertical Movement (Gravity & Jump) ---
        if movement == 1 and self.on_ground:  # Jump
            self.player_vel.y = self.JUMP_FORCE
            self.on_ground = False
            for _ in range(10):
                self._create_particle(self.player_rect.midbottom, life=15, speed_range=(1, 4))


        self.player_vel.y += self.GRAVITY
        if self.player_vel.y > self.TERMINAL_VELOCITY:
            self.player_vel.y = self.TERMINAL_VELOCITY

        # --- 3. Update Positions & Handle Collisions ---
        self.player_rect.x += int(self.player_vel.x)
        
        # Update entity states (moving/collapsing platforms)
        self._update_platforms()

        # Vertical movement and collision
        self.player_rect.y += int(self.player_vel.y)
        self.on_ground = False
        
        for plat in self.platforms:
            if self.player_rect.colliderect(plat['rect']):
                # Check for vertical collision (landing on top)
                if self.player_vel.y > 0 and self.player_rect.bottom - self.player_vel.y <= plat['rect'].top:
                    self.player_rect.bottom = plat['rect'].top
                    self.player_vel.y = 0
                    if not self.on_ground: # First frame of landing
                        for _ in range(5):
                            self._create_particle(self.player_rect.midbottom, life=10, speed_range=(0.5, 2))
                    self.on_ground = True
                    
                    # Handle platform specific logic
                    if plat['type'] == 'moving':
                        self.player_rect.x += plat['vel'].x
                    elif plat['type'] == 'collapsing' and plat['state'] == 'stable':
                        plat['state'] = 'shaking'
                        plat['timer'] = 15 # Start collapse timer
                        if not plat.get('reward_given', False):
                            reward += 10 # Reward for landing on risky platform
                            plat['reward_given'] = True


        # --- 4. Handle other interactions ---
        # Coin collection
        collected_coins = []
        for coin in self.coins:
            if self.player_rect.colliderect(coin):
                collected_coins.append(coin)
                self.score += 1
                reward += 1
                for _ in range(15):
                    self._create_particle(coin.center, life=20, speed_range=(2,5), color=self.COLOR_COIN)
        self.coins = [c for c in self.coins if c not in collected_coins]

        # Reward for forward progress
        progress = self.player_rect.x - self.max_dist_reached
        if progress > 0:
            reward += progress * 0.1
            self.max_dist_reached = self.player_rect.x

        # --- 5. Update Game State ---
        self.steps += 1
        self.time_left -= 1
        self._update_difficulty()

        # --- 6. Check for Termination ---
        terminated = False
        truncated = False
        if self.player_rect.top > self.HEIGHT:  # Fell off screen
            terminated = True
            reward -= 5
        elif self.player_rect.colliderect(self.flag_rect): # Reached flag
            terminated = True
            reward += 100
            self.score += 50 # Bonus for finishing
        elif self.time_left <= 0: # Time out
            terminated = True
            truncated = True
        elif self.steps >= self.MAX_STEPS: # Step limit
            terminated = True
            truncated = True

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info(),
        )

    def _generate_level(self):
        self.platforms = []
        self.coins = []
        
        start_plat = {'rect': pygame.Rect(50, self.HEIGHT - 40, 150, 40), 'type': 'static', 'color': self.COLOR_STATIC_PLATFORM}
        self.platforms.append(start_plat)
        
        current_x = start_plat['rect'].right
        current_y = start_plat['rect'].top
        level_length = 5000

        max_jump_dist = abs(self.PLAYER_SPEED * (2 * self.JUMP_FORCE / self.GRAVITY)) * 0.9
        
        while current_x < level_length:
            gap = self.np_random.uniform(30, max_jump_dist - self.PLAYER_SIZE) + self.gap_increase
            width = self.np_random.uniform(80, 200)
            y_change = self.np_random.uniform(-100, 100)
            
            x = current_x + gap
            y = np.clip(current_y + y_change, 150, self.HEIGHT - 40)
            
            plat_type_roll = self.np_random.random()
            if plat_type_roll < 0.65:
                plat = {'rect': pygame.Rect(x, y, width, 20), 'type': 'static', 'color': self.COLOR_STATIC_PLATFORM}
            elif plat_type_roll < 0.85:
                move_speed = self.np_random.uniform(1, 2.5) + self.speed_increase
                move_range = self.np_random.uniform(50, 100)
                plat = {
                    'rect': pygame.Rect(x, y, width, 20), 'type': 'moving', 'color': self.COLOR_MOVING_PLATFORM,
                    'start_pos': pygame.Vector2(x, y), 'vel': pygame.Vector2(move_speed, 0), 'range': move_range
                }
            else:
                plat = {
                    'rect': pygame.Rect(x, y, width, 20), 'type': 'collapsing', 'color': self.COLOR_COLLAPSING_PLATFORM,
                    'state': 'stable', 'timer': 0
                }
            self.platforms.append(plat)
            
            for i in range(self.np_random.integers(1, 4)):
                coin_x = x + (i + 1) * (width / 4)
                coin_y = y - self.np_random.uniform(40, 80)
                self.coins.append(pygame.Rect(coin_x, coin_y, 16, 16))

            current_x = x + width
            current_y = y
        
        self.level_end_x = current_x + 100
        self.flag_rect = pygame.Rect(self.level_end_x, current_y - 80, 20, 80)

    def _update_platforms(self):
        for plat in self.platforms:
            if plat['type'] == 'moving':
                plat['rect'].move_ip(plat['vel'])
                if abs(plat['rect'].x - plat['start_pos'].x) > plat['range']:
                    plat['vel'] *= -1
            elif plat['type'] == 'collapsing' and plat['state'] != 'stable':
                if plat['state'] == 'shaking':
                    plat['timer'] -= 1
                    if plat['timer'] <= 0:
                        plat['state'] = 'falling'
                        plat['fall_speed'] = 2
                elif plat['state'] == 'falling':
                    plat['fall_speed'] += self.GRAVITY / 2
                    plat['rect'].y += int(plat['fall_speed'])
        
        self.platforms = [p for p in self.platforms if p['rect'].top < self.HEIGHT + 50]

    def _update_difficulty(self):
        if self.steps > 0 and self.steps % 500 == 0:
            self.gap_increase += 2
            self.speed_increase += 0.05

    def _get_observation(self):
        target_camera_x = self.player_rect.x - self.WIDTH / 3
        self.camera_x += (target_camera_x - self.camera_x) * 0.1

        self.screen.fill(self.COLOR_BG)
        self._render_background_scenery()
        self._render_platforms()
        self._render_coins()
        self._render_flag()
        self._render_player()
        self._render_particles()
        self._render_ui()

        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "time_left": int(self.time_left / self.FPS),
            "player_pos": (self.player_rect.x, self.player_rect.y)
        }

    def _render_text(self, text, font, color, pos, shadow=True):
        if shadow:
            text_surface_shadow = font.render(text, True, self.COLOR_TEXT_SHADOW)
            self.screen.blit(text_surface_shadow, (pos[0] + 2, pos[1] + 2))
        text_surface = font.render(text, True, color)
        self.screen.blit(text_surface, pos)

    def _render_background_scenery(self):
        for i in range(5):
            cloud_x = 100 + i * 250 - (self.camera_x * 0.2) % (250 * 5)
            cloud_y = 50 + (i % 2) * 30
            pygame.gfxdraw.filled_ellipse(self.screen, int(cloud_x), int(cloud_y), 40, 20, (255, 255, 255, 100))
            pygame.gfxdraw.filled_ellipse(self.screen, int(cloud_x+30), int(cloud_y+10), 50, 25, (255, 255, 255, 100))

    def _render_platforms(self):
        for plat in self.platforms:
            render_rect = plat['rect'].move(-int(self.camera_x), 0)
            if self.screen.get_rect().colliderect(render_rect):
                color = plat['color']
                if plat['type'] == 'collapsing' and plat['state'] == 'shaking':
                    offset_x = self.np_random.uniform(-2, 2)
                    offset_y = self.np_random.uniform(-2, 2)
                    render_rect.move_ip(offset_x, offset_y)
                    alpha = max(0, 255 * (plat['timer'] / 15.0))
                    # Create a temporary surface for alpha blending
                    temp_surf = pygame.Surface(render_rect.size, pygame.SRCALPHA)
                    temp_surf.fill((*color, alpha))
                    self.screen.blit(temp_surf, render_rect.topleft)
                else:
                    pygame.draw.rect(self.screen, color, render_rect, border_radius=3)
                
                # Add a darker top edge for depth
                top_edge = pygame.Rect(render_rect.left, render_rect.top, render_rect.width, 4)
                # FIX: Convert generator to a tuple for the color argument
                darker_color = tuple(max(0, c - 30) for c in color[:3])
                pygame.draw.rect(self.screen, darker_color, top_edge, border_top_left_radius=3, border_top_right_radius=3)

    def _render_coins(self):
        for coin in self.coins:
            render_rect = coin.move(-int(self.camera_x), 0)
            if self.screen.get_rect().colliderect(render_rect):
                anim_offset = math.sin((self.steps + coin.x) * 0.1)
                render_rect.y += int(anim_offset * 4)
                spin_width = int(abs(anim_offset) * coin.width)
                spin_rect = pygame.Rect(0, 0, max(2, spin_width), coin.height)
                spin_rect.center = render_rect.center
                pygame.draw.ellipse(self.screen, self.COLOR_COIN, spin_rect)
                pygame.draw.ellipse(self.screen, (255,255,150), spin_rect, 2)

    def _render_flag(self):
        render_pole = self.flag_rect.move(-int(self.camera_x), 0)
        pygame.draw.rect(self.screen, (192, 192, 192), render_pole)
        
        flag_points = [
            (render_pole.right, render_pole.top),
            (render_pole.right + 50, render_pole.top + 15 + math.sin(self.steps*0.2)*5),
            (render_pole.right, render_pole.top + 30)
        ]
        pygame.draw.polygon(self.screen, self.COLOR_FLAG, flag_points)

    def _render_player(self):
        render_rect = self.player_rect.move(-int(self.camera_x), 0)
        
        if self.on_ground and abs(self.player_vel.x) > 0:
            stretch = abs(math.sin(self.steps * 0.5)) * 4
            render_rect.height -= stretch
            render_rect.y += stretch
        elif not self.on_ground:
            stretch = min(6, int(abs(self.player_vel.y) * 0.5))
            render_rect.height += stretch
            render_rect.width -= stretch
            render_rect.x += stretch // 2
        
        glow_rect = render_rect.inflate(8, 8)
        glow_surf = pygame.Surface(glow_rect.size, pygame.SRCALPHA)
        pygame.draw.ellipse(glow_surf, (*self.COLOR_PLAYER_GLOW, 50), glow_surf.get_rect())
        self.screen.blit(glow_surf, glow_rect.topleft)

        pygame.draw.rect(self.screen, self.COLOR_PLAYER, render_rect, border_radius=4)
    
    def _create_particle(self, pos, life, speed_range, color=None):
        angle = self.np_random.uniform(0, 2 * math.pi)
        speed = self.np_random.uniform(*speed_range)
        vel = pygame.Vector2(math.cos(angle) * speed, math.sin(angle) * speed)
        particle = {
            'pos': pygame.Vector2(pos),
            'vel': vel,
            'life': life,
            'max_life': life,
            'color': color or self.COLOR_PARTICLE
        }
        self.particles.append(particle)
        
    def _render_particles(self):
        for p in list(self.particles):
            p['pos'] += p['vel']
            p['life'] -= 1
            if p['life'] <= 0:
                if p in self.particles:
                    self.particles.remove(p)
                continue
            
            alpha = int(255 * (p['life'] / p['max_life']))
            color = (*p['color'][:3], alpha)
            size = int(5 * (p['life'] / p['max_life']))
            
            pos = (int(p['pos'].x - self.camera_x), int(p['pos'].y))
            
            # Use gfxdraw for alpha blending
            if size > 0:
                pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], max(1, size), color)

    def _render_ui(self):
        score_text = f"SCORE: {self.score}"
        self._render_text(score_text, self.font_small, self.COLOR_TEXT, (10, 10))
        
        time_str = f"TIME: {max(0, int(self.time_left / self.FPS))}"
        time_color = self.COLOR_TEXT if self.time_left > 10 * self.FPS else (255, 0, 0)
        time_size = self.font_small.size(time_str)
        self._render_text(time_str, self.font_small, time_color, (self.WIDTH - time_size[0] - 10, 10))

    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # To run with a display, set the following environment variable
    # and install the required libraries (e.g., `pip install pygame gymnasium`)
    # For linux: os.environ['SDL_VIDEODRIVER'] = 'x11'
    # For windows: os.environ['SDL_VIDEODRIVER'] = 'windows'
    
    # Set to a display-compatible driver to view the game
    try:
        os.environ['SDL_VIDEODRIVER'] = 'x11'
        pygame.display.init()
        pygame.font.init()
    except pygame.error:
        print("No display available, running in headless mode.")
        os.environ['SDL_VIDEODRIVER'] = 'dummy'

    
    env = GameEnv(render_mode="rgb_array")
    
    # --- Manual Play ---
    # This setup allows a human to play the game if a display is available
    use_display = 'dummy' not in os.environ.get('SDL_VIDEODRIVER', 'dummy')
    if use_display:
        pygame.display.set_caption("Pixel Platformer")
        screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    
    obs, info = env.reset(seed=42)
    terminated = False
    truncated = False
    
    action = [0, 0, 0] 
    
    while not (terminated or truncated):
        if use_display:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    terminated = True
            
            keys = pygame.key.get_pressed()
            
            if keys[pygame.K_UP]: action[0] = 1
            elif keys[pygame.K_DOWN]: action[0] = 2
            elif keys[pygame.K_LEFT]: action[0] = 3
            elif keys[pygame.K_RIGHT]: action[0] = 4
            else: action[0] = 0
            
            action[1] = 1 if keys[pygame.K_SPACE] else 0
            action[2] = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        else: # Simple agent for headless mode
            action = env.action_space.sample()

        obs, reward, term, trunc, info = env.step(action)
        terminated = term
        truncated = trunc
        
        if use_display:
            surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
            screen.blit(surf, (0, 0))
            pygame.display.flip()
        
        env.clock.tick(env.FPS)
        
        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']}, Steps: {info['steps']}")
            obs, info = env.reset(seed=43)
            terminated, truncated = False, False

    env.close()