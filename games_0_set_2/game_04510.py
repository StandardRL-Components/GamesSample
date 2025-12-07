
# Generated: 2025-08-28T02:37:44.180892
# Source Brief: brief_04510.md
# Brief Index: 4510

        
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
        "Controls: ↑ to jump, ←→ to move. Collect all the carrots to win!"
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A cheerful side-scrolling arcade game. Help the bunny collect carrots while hopping over rolling logs."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    GROUND_LEVEL = 360
    MAX_STEPS = 2000
    WIN_SCORE = 20

    # Colors
    COLOR_SKY = (135, 206, 235)
    COLOR_GROUND = (34, 139, 34)
    COLOR_HILL_FAR = (107, 142, 35)
    COLOR_HILL_NEAR = (85, 107, 47)
    COLOR_BUNNY = (255, 255, 255)
    COLOR_BUNNY_OUTLINE = (100, 100, 100)
    COLOR_BUNNY_EYE = (0, 0, 0)
    COLOR_BUNNY_NOSE = (255, 182, 193)
    COLOR_CARROT = (255, 140, 0)
    COLOR_CARROT_TOP = (0, 128, 0)
    COLOR_OBSTACLE = (139, 69, 19)
    COLOR_OBSTACLE_SPIKE = (92, 64, 51)
    COLOR_UI_TEXT = (255, 255, 255)
    COLOR_UI_SHADOW = (0, 0, 0)
    COLOR_UI_BG = (0, 0, 0, 128)

    # Physics & Gameplay
    GRAVITY = 0.6
    JUMP_STRENGTH = 13
    PLAYER_SPEED = 5
    INITIAL_OBSTACLE_SPEED = 2.0
    OBSTACLE_SPEED_INCREASE = 0.1
    CAMERA_SMOOTHING = 0.08
    
    # Rewards
    REWARD_SURVIVE = 0.01
    REWARD_CARROT = 10.0
    REWARD_WIN = 100.0
    REWARD_FAIL = -100.0

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
        
        self.font_main = pygame.font.Font(None, 48)
        self.font_ui = pygame.font.Font(None, 36)

        # Initialize state variables (these will be properly set in reset)
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False
        
        self.player_pos = [0, 0]
        self.player_vel_y = 0
        self.player_on_ground = True
        self.bunny_squash = 1.0
        self.bunny_ear_angle = 0.0
        
        self.camera_offset_x = 0.0
        self.smooth_camera_offset_x = 0.0
        self.obstacle_speed = self.INITIAL_OBSTACLE_SPEED
        
        self.obstacles = []
        self.carrots = []
        self.particles = []
        self.background_hills = []
        
        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False
        
        self.player_pos = [self.SCREEN_WIDTH / 2, self.GROUND_LEVEL]
        self.player_vel_y = 0
        self.player_on_ground = True
        self.bunny_squash = 1.0
        self.bunny_ear_angle = 0.0
        
        self.camera_offset_x = self.player_pos[0] - self.SCREEN_WIDTH / 2
        self.smooth_camera_offset_x = self.camera_offset_x
        self.obstacle_speed = self.INITIAL_OBSTACLE_SPEED
        
        self.obstacles.clear()
        self.carrots.clear()
        self.particles.clear()
        self.background_hills.clear()
        
        self._spawn_initial_entities()
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        reward = 0
        terminated = False

        if not self.game_over:
            self._handle_input(action)
            self._update_game_state()
            reward = self._process_collisions_and_rewards()
        else:
            self._update_particles() # Keep particles moving after game over

        self.steps += 1
        
        if self.score >= self.WIN_SCORE and not self.game_over:
            self.win = True
            terminated = True
            reward += self.REWARD_WIN
        
        if self.steps >= self.MAX_STEPS:
            terminated = True

        if self.game_over:
            terminated = True
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_input(self, action):
        movement = action[0]

        if movement == 3:  # Left
            self.player_pos[0] -= self.PLAYER_SPEED
            if self.player_on_ground:
                self._create_dust_particles(self.player_pos[0] + 15, 5)
        elif movement == 4:  # Right
            self.player_pos[0] += self.PLAYER_SPEED
            if self.player_on_ground:
                self._create_dust_particles(self.player_pos[0] - 15, 5)

        if movement == 1 and self.player_on_ground:  # Jump
            self.player_vel_y = -self.JUMP_STRENGTH
            self.player_on_ground = False
            self.bunny_squash = 1.5 # Stretch for jump
            # sfx: jump

    def _update_game_state(self):
        # --- Player Physics ---
        self.player_vel_y += self.GRAVITY
        self.player_pos[1] += self.player_vel_y

        if self.player_pos[1] >= self.GROUND_LEVEL:
            if not self.player_on_ground: # Just landed
                self.bunny_squash = 0.7 # Squash on land
                self._create_dust_particles(self.player_pos[0], 10)
                # sfx: land
            self.player_pos[1] = self.GROUND_LEVEL
            self.player_vel_y = 0
            self.player_on_ground = True
        
        # --- Animations ---
        self.bunny_squash += (1.0 - self.bunny_squash) * 0.15 # Lerp back to 1.0
        self.bunny_ear_angle = -self.player_vel_y * 1.5 # Ears flap based on vertical velocity

        # --- Camera ---
        target_camera_offset = self.player_pos[0] - self.SCREEN_WIDTH / 2
        self.smooth_camera_offset_x += (target_camera_offset - self.smooth_camera_offset_x) * self.CAMERA_SMOOTHING

        # --- Entities ---
        self._update_obstacles()
        self._update_particles()
        self._spawn_entities()

    def _process_collisions_and_rewards(self):
        reward = self.REWARD_SURVIVE
        player_rect = pygame.Rect(self.player_pos[0] - 15, self.player_pos[1] - 40, 30, 40)

        # Carrot collection
        for carrot in self.carrots[:]:
            carrot_rect = pygame.Rect(carrot['x'] - 10, carrot['y'] - 15, 20, 30)
            if player_rect.colliderect(carrot_rect):
                self.carrots.remove(carrot)
                self.score += 1
                reward += self.REWARD_CARROT
                self._create_collect_particles(carrot['x'], carrot['y'])
                # sfx: collect_carrot
                
                # Difficulty scaling
                if self.score > 0 and self.score % 5 == 0:
                    self.obstacle_speed += self.OBSTACLE_SPEED_INCREASE

        # Obstacle collision
        for obstacle in self.obstacles:
            # Simple circle collision
            dist_sq = (self.player_pos[0] - obstacle['x'])**2 + (self.player_pos[1] - 20 - obstacle['y'])**2
            if dist_sq < (15 + obstacle['radius'])**2:
                self.game_over = True
                reward = self.REWARD_FAIL # Overwrite other rewards
                self._create_death_particles(self.player_pos[0], self.player_pos[1] - 20)
                # sfx: player_hit
                break
        
        return reward

    def _spawn_initial_entities(self):
        for _ in range(15): # More hills for a dense background
            self._spawn_hill(random_side=True)
        for _ in range(10):
            self._spawn_carrot(random_pos=True)
        for _ in range(3):
            self._spawn_obstacle(random_pos=True)
        
        self.carrots.sort(key=lambda c: c['x'])
        self.obstacles.sort(key=lambda o: o['x'])

    def _spawn_entities(self):
        # Spawn hills to keep background populated
        if self.np_random.random() < 0.1:
            self._spawn_hill()

        # Spawn carrots and obstacles off-screen
        spawn_roll = self.np_random.random()
        if spawn_roll < 0.02: # 2% chance per frame to spawn a carrot
            self._spawn_carrot()
        elif spawn_roll < 0.025: # 0.5% chance per frame to spawn an obstacle
            self._spawn_obstacle()

    def _spawn_hill(self, random_side=False):
        parallax = self.np_random.uniform(0.2, 0.6)
        radius = self.np_random.integers(100, 400)
        y = self.GROUND_LEVEL + radius * 0.7
        color = self.COLOR_HILL_NEAR if parallax > 0.4 else self.COLOR_HILL_FAR
        
        if random_side:
             x = self.smooth_camera_offset_x + self.np_random.uniform(-self.SCREEN_WIDTH, self.SCREEN_WIDTH * 2)
        else:
            side = 1 if self.np_random.random() < 0.5 else -1
            x = self.smooth_camera_offset_x + side * (self.SCREEN_WIDTH / 2 + radius) + self.np_random.uniform(50, 200) * side
        
        self.background_hills.append({'x': x, 'y': y, 'radius': radius, 'color': color, 'parallax': parallax})
        # Keep list from growing infinitely
        if len(self.background_hills) > 50:
            self.background_hills.pop(0)

    def _spawn_carrot(self, random_pos=False):
        if random_pos:
            x = self.np_random.uniform(self.SCREEN_WIDTH, self.SCREEN_WIDTH * 3)
        else:
            x = self.smooth_camera_offset_x + self.SCREEN_WIDTH + self.np_random.uniform(50, 150)
        
        y = self.np_random.uniform(self.GROUND_LEVEL - 100, self.GROUND_LEVEL - 20)
        self.carrots.append({'x': x, 'y': y})

    def _spawn_obstacle(self, random_pos=False):
        if random_pos:
            x = self.np_random.uniform(self.SCREEN_WIDTH * 1.5, self.SCREEN_WIDTH * 4)
        else:
            x = self.smooth_camera_offset_x + self.SCREEN_WIDTH + self.np_random.uniform(200, 400)
        
        self.obstacles.append({
            'x': x, 
            'y': self.GROUND_LEVEL, 
            'radius': self.np_random.integers(20, 35),
            'rot': self.np_random.uniform(0, 360)
        })

    def _update_obstacles(self):
        for o in self.obstacles:
            o['x'] -= self.obstacle_speed
            o['rot'] -= self.obstacle_speed * 1.5
        
        # Remove off-screen obstacles
        self.obstacles = [o for o in self.obstacles if o['x'] > self.smooth_camera_offset_x - 100]

    def _update_particles(self):
        for p in self.particles[:]:
            p['x'] += p['vx']
            p['y'] += p['vy']
            p['vy'] += 0.1 # a little gravity
            p['life'] -= 1
            if p['life'] <= 0:
                self.particles.remove(p)
    
    def _create_particle(self, x, y, color, count, speed, life):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            s = self.np_random.uniform(0.5, 1.0) * speed
            self.particles.append({
                'x': x, 'y': y,
                'vx': math.cos(angle) * s, 'vy': math.sin(angle) * s,
                'life': self.np_random.integers(life // 2, life),
                'color': color,
                'size': self.np_random.integers(2, 5)
            })

    def _create_dust_particles(self, x, count):
        for _ in range(count):
            self.particles.append({
                'x': x + self.np_random.uniform(-5, 5), 
                'y': self.GROUND_LEVEL + self.np_random.uniform(0, 5),
                'vx': self.np_random.uniform(-0.5, 0.5), 
                'vy': self.np_random.uniform(-1, -0.2),
                'life': self.np_random.integers(15, 30),
                'color': (188, 143, 143, 150),
                'size': self.np_random.integers(3, 7)
            })

    def _create_collect_particles(self, x, y):
        self._create_particle(x, y, (255, 215, 0), 15, 3, 30)

    def _create_death_particles(self, x, y):
        self._create_particle(x, y, self.COLOR_BUNNY, 30, 4, 50)
        self._create_particle(x, y, self.COLOR_BUNNY_NOSE, 5, 4, 50)

    def _get_observation(self):
        # --- Background ---
        self.screen.fill(self.COLOR_SKY)
        
        # Parallax Hills
        self.background_hills.sort(key=lambda h: h['parallax'])
        for hill in self.background_hills:
            draw_x = int(hill['x'] - self.smooth_camera_offset_x * hill['parallax'])
            pygame.gfxdraw.filled_circle(self.screen, draw_x, int(hill['y']), int(hill['radius']), hill['color'])
        
        # Ground
        pygame.draw.rect(self.screen, self.COLOR_GROUND, (0, self.GROUND_LEVEL, self.SCREEN_WIDTH, self.SCREEN_HEIGHT - self.GROUND_LEVEL))

        # --- Game Objects ---
        cam_x = self.smooth_camera_offset_x

        # Particles
        for p in self.particles:
            pygame.draw.circle(self.screen, p['color'], (int(p['x'] - cam_x), int(p['y'])), int(p['size'] * (p['life'] / 30.0)))
        
        # Carrots
        for carrot in self.carrots:
            self._draw_carrot(self.screen, int(carrot['x'] - cam_x), int(carrot['y']))

        # Obstacles
        for obstacle in self.obstacles:
            self._draw_obstacle(self.screen, int(obstacle['x'] - cam_x), int(obstacle['y']), obstacle['radius'], obstacle['rot'])

        # Player (Bunny)
        if not self.game_over:
            self._draw_bunny(self.screen, int(self.player_pos[0] - cam_x), int(self.player_pos[1]))
        
        # --- UI ---
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _draw_bunny(self, surface, x, y):
        body_h = 40 * self.bunny_squash
        body_w = 30 / self.bunny_squash
        body_y = y - body_h

        # Ears
        ear_h, ear_w = 30, 8
        ear_angle_rad = math.radians(self.bunny_ear_angle)
        for i in [-1, 1]:
            ear_x = x + i * (body_w / 3)
            ear_y = body_y
            ear_poly = [
                (ear_x - ear_w/2, ear_y),
                (ear_x + ear_w/2, ear_y),
                (ear_x + ear_w/2, ear_y - ear_h),
                (ear_x - ear_w/2, ear_y - ear_h),
            ]
            # Rotate ears
            rotated_poly = []
            for px, py in ear_poly:
                tx, ty = px - ear_x, py - ear_y
                rx = tx * math.cos(ear_angle_rad) - ty * math.sin(ear_angle_rad)
                ry = tx * math.sin(ear_angle_rad) + ty * math.cos(ear_angle_rad)
                rotated_poly.append((rx + ear_x, ry + ear_y))

            pygame.gfxdraw.filled_polygon(surface, rotated_poly, self.COLOR_BUNNY)
            pygame.gfxdraw.aapolygon(surface, rotated_poly, self.COLOR_BUNNY_OUTLINE)

        # Body
        body_rect = pygame.Rect(x - body_w/2, body_y, body_w, body_h)
        pygame.draw.ellipse(surface, self.COLOR_BUNNY, body_rect)
        pygame.draw.ellipse(surface, self.COLOR_BUNNY_OUTLINE, body_rect, 2)
        
        # Face
        eye_x = x + body_w / 4
        eye_y = body_y + body_h / 3
        pygame.draw.circle(surface, self.COLOR_BUNNY_EYE, (int(eye_x), int(eye_y)), 2)
        nose_x = x + body_w / 3
        nose_y = body_y + body_h / 2
        pygame.draw.circle(surface, self.COLOR_BUNNY_NOSE, (int(nose_x), int(nose_y)), 3)

    def _draw_carrot(self, surface, x, y):
        # Carrot body
        points = [(x, y), (x - 10, y - 25), (x + 10, y - 25)]
        pygame.gfxdraw.filled_trigon(surface, points[0][0], points[0][1], points[1][0], points[1][1], points[2][0], points[2][1], self.COLOR_CARROT)
        pygame.gfxdraw.aatrigon(surface, points[0][0], points[0][1], points[1][0], points[1][1], points[2][0], points[2][1], self.COLOR_CARROT)
        # Carrot top
        pygame.draw.rect(surface, self.COLOR_CARROT_TOP, (x - 5, y - 30, 10, 5))

    def _draw_obstacle(self, surface, x, y, radius, rot):
        # Main log body
        pygame.gfxdraw.filled_circle(surface, x, int(y - radius), radius, self.COLOR_OBSTACLE)
        pygame.gfxdraw.aacircle(surface, x, int(y - radius), radius, self.COLOR_OBSTACLE_SPIKE)
        # Spikes
        for i in range(8):
            angle = math.radians(rot + i * 45)
            p1 = (x, y - radius)
            p2 = (x + math.cos(angle) * radius, y - radius + math.sin(angle) * radius)
            p3 = (x + math.cos(angle + 0.2) * (radius + 10), y - radius + math.sin(angle + 0.2) * (radius + 10))
            p4 = (x + math.cos(angle - 0.2) * (radius + 10), y - radius + math.sin(angle - 0.2) * (radius + 10))
            pygame.gfxdraw.filled_polygon(surface, [p2,p3,p4], self.COLOR_OBSTACLE_SPIKE)

    def _render_ui(self):
        # Score
        carrot_icon_surf = pygame.Surface((30, 30), pygame.SRCALPHA)
        self._draw_carrot(carrot_icon_surf, 15, 28)
        self.screen.blit(carrot_icon_surf, (10, 10))
        
        score_text = f"x {self.score}"
        self._draw_text(score_text, self.font_ui, (50, 15), self.COLOR_UI_TEXT, self.COLOR_UI_SHADOW)

        # Game Over / Win message
        if self.game_over or self.win:
            message = "YOU WIN!" if self.win else "GAME OVER"
            s = pygame.Surface((self.SCREEN_WIDTH, 100), pygame.SRCALPHA)
            s.fill(self.COLOR_UI_BG)
            self.screen.blit(s, (0, self.SCREEN_HEIGHT / 2 - 50))
            self._draw_text(message, self.font_main, (self.SCREEN_WIDTH/2, self.SCREEN_HEIGHT/2), self.COLOR_UI_TEXT, self.COLOR_UI_SHADOW, center=True)

    def _draw_text(self, text, font, pos, color, shadow_color, center=False):
        text_surf = font.render(text, True, color)
        text_shadow_surf = font.render(text, True, shadow_color)
        text_rect = text_surf.get_rect()
        if center:
            text_rect.center = pos
        else:
            text_rect.topleft = pos
        
        self.screen.blit(text_shadow_surf, (text_rect.x + 2, text_rect.y + 2))
        self.screen.blit(text_surf, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
        }

    def validate_implementation(self):
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
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game directly
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Bunny Hop")
    clock = pygame.time.Clock()
    
    terminated = False
    running = True
    
    action = env.action_space.sample()
    action.fill(0) # Start with no-op

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        if not terminated:
            keys = pygame.key.get_pressed()
            
            # Reset action
            action.fill(0)
            
            # Movement
            if keys[pygame.K_UP]:
                action[0] = 1
            elif keys[pygame.K_DOWN]:
                action[0] = 2
            elif keys[pygame.K_LEFT]:
                action[0] = 3
            elif keys[pygame.K_RIGHT]:
                action[0] = 4
            
            # Other buttons
            if keys[pygame.K_SPACE]:
                action[1] = 1
            if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
                action[2] = 1

            obs, reward, terminated, truncated, info = env.step(action)
            
            if terminated:
                print(f"Episode finished. Score: {info['score']}, Steps: {info['steps']}")

        # Render the observation from the environment
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        clock.tick(30) # Run at 30 FPS

    pygame.quit()