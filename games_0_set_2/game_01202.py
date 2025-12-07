
# Generated: 2025-08-27T16:21:24.351400
# Source Brief: brief_01202.md
# Brief Index: 1202

        
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

    user_guide = (
        "Controls: Press space for a small jump, or hold shift for a large jump. Avoid the asteroids!"
    )

    game_description = (
        "Guide a hopping spaceship through obstacle-filled space, maximizing distance and surviving to reach the end of three stages."
    )

    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.GROUND_Y = self.HEIGHT - 50
        self.FPS = 30

        # Colors
        self.COLOR_BG = (15, 15, 25)
        self.COLOR_STARS = [(50, 50, 60), (100, 100, 120), (200, 200, 220)]
        self.COLOR_GROUND = (60, 180, 75)
        self.COLOR_PLAYER = (50, 150, 255)
        self.COLOR_PLAYER_GLOW = (150, 200, 255)
        self.COLOR_OBSTACLE = (200, 70, 70)
        self.COLOR_OBSTACLE_OUTLINE = (120, 40, 40)
        self.COLOR_BONUS = (255, 220, 0)
        self.COLOR_TEXT = (255, 255, 255)
        self.COLOR_TIMER_WARN = (255, 100, 100)

        # Physics & Gameplay
        self.GRAVITY = 0.5
        self.SMALL_JUMP_VEL = -7.5
        self.LARGE_JUMP_VEL = -11.0
        self.PLAYER_X_POS = 100
        self.PLAYER_RADIUS = 12
        self.BASE_SCROLL_SPEED = 3.0
        self.STAGE_LENGTH = 4800  # Distance units to complete a stage
        self.MAX_STAGES = 3
        self.TIME_PER_STAGE_S = 60
        self.TIME_PER_STAGE_STEPS = self.TIME_PER_STAGE_S * self.FPS
        self.INITIAL_LIVES = 3

        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("monospace", 20, bold=True)
        self.font_game_over = pygame.font.SysFont("monospace", 48, bold=True)
        
        # --- Game State ---
        # These are initialized in reset()
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.victory = False
        self.np_random = None
        self.player_pos = None
        self.player_vel = None
        self.on_ground = None
        self.camera_x = None
        self.scroll_speed = None
        self.obstacles = None
        self.bonuses = None
        self.particles = None
        self.lives = None
        self.stage = None
        self.stage_timer = None
        self.stars = None
        self.last_jump_unnecessary = False
        self.next_obstacle_spawn_dist = 0
        self.next_bonus_spawn_dist = 0

        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.victory = False

        self.player_pos = np.array([float(self.PLAYER_X_POS), float(self.GROUND_Y)])
        self.player_vel = np.array([0.0, 0.0])
        self.on_ground = True
        
        self.camera_x = 0.0
        self.scroll_speed = self.BASE_SCROLL_SPEED
        
        self.obstacles = []
        self.bonuses = []
        self.particles = []
        
        self.lives = self.INITIAL_LIVES
        self.stage = 1
        self.stage_timer = 0
        
        self.last_jump_unnecessary = False
        self.next_obstacle_spawn_dist = 0
        self.next_bonus_spawn_dist = 0

        self._generate_initial_stars()

        return self._get_observation(), self._get_info()

    def step(self, action):
        reward = 0
        self.last_jump_unnecessary = False
        
        if not self.game_over:
            # --- Handle Input ---
            self._handle_input(action)

            # --- Update Physics & Game State ---
            self._update_player_physics()
            self._update_world()
            self._update_particles()
            
            # --- Handle Collisions & Events ---
            collision_reward = self._handle_collisions()
            reward += collision_reward

            # --- Update Timers and Stage ---
            self.stage_timer += 1
            if self.camera_x >= self.stage * self.STAGE_LENGTH:
                self._advance_stage()

        # --- Calculate Reward ---
        # Reward for forward progress
        reward += self.scroll_speed * 0.01 
        # Penalty for unnecessary "safe" jumps
        if self.last_jump_unnecessary:
            reward -= 0.2
        
        self.score += reward

        # --- Check Termination ---
        terminated = self._check_termination()

        # Tick clock for auto-advance
        if self.auto_advance:
            self.clock.tick(self.FPS)

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info(),
        )

    def _handle_input(self, action):
        space_held = action[1] == 1
        shift_held = action[2] == 1

        if self.on_ground:
            jump_vel = 0
            if shift_held:
                jump_vel = self.LARGE_JUMP_VEL
            elif space_held:
                jump_vel = self.SMALL_JUMP_VEL
            
            if jump_vel != 0:
                self.player_vel[1] = jump_vel
                self.on_ground = False
                # sfx: jump_sound()
                self._create_particles(self.player_pos + np.array([0, self.PLAYER_RADIUS]), 20, self.COLOR_PLAYER, 'thrust')

                # Check if jump was "unnecessary"
                safe_to_jump = True
                for obs in self.obstacles:
                    if obs['rect'].left < self.player_pos[0] + 300:
                        safe_to_jump = False
                        break
                if safe_to_jump:
                    self.last_jump_unnecessary = True


    def _update_player_physics(self):
        if not self.on_ground:
            self.player_vel[1] += self.GRAVITY
            self.player_pos += self.player_vel

        if self.player_pos[1] >= self.GROUND_Y:
            self.player_pos[1] = self.GROUND_Y
            self.player_vel[1] = 0
            if not self.on_ground:
                # sfx: land_sound()
                self._create_particles(self.player_pos + np.array([0, self.PLAYER_RADIUS-5]), 10, self.COLOR_GROUND, 'land')
            self.on_ground = True
            
        # Keep player within screen bounds (top)
        if self.player_pos[1] < self.PLAYER_RADIUS:
            self.player_pos[1] = self.PLAYER_RADIUS
            self.player_vel[1] = 0


    def _update_world(self):
        self.camera_x += self.scroll_speed

        # --- Spawn new entities ---
        if self.camera_x > self.next_obstacle_spawn_dist:
            self._spawn_obstacle()
        
        if self.camera_x > self.next_bonus_spawn_dist:
            self._spawn_bonus()

        # --- Update and cull entities ---
        offscreen_margin = 100
        self.obstacles = [o for o in self.obstacles if o['world_x'] > self.camera_x - offscreen_margin]
        self.bonuses = [b for b in self.bonuses if b['world_x'] > self.camera_x - offscreen_margin]

    def _spawn_obstacle(self):
        obs_world_x = self.camera_x + self.WIDTH + 50
        
        # Vary height and size
        height_choice = self.np_random.choice(['ground', 'low_air', 'high_air'])
        if height_choice == 'ground':
            w = self.np_random.integers(40, 80)
            h = self.np_random.integers(30, 60)
            y = self.GROUND_Y - h
        elif height_choice == 'low_air':
            w = self.np_random.integers(50, 100)
            h = self.np_random.integers(40, 70)
            y = self.np_random.integers(self.GROUND_Y - 150, self.GROUND_Y - h - 20)
        else: # high_air
            w = self.np_random.integers(60, 120)
            h = self.np_random.integers(50, 80)
            y = self.np_random.integers(50, self.GROUND_Y - 220)

        rect = pygame.Rect(0, y, w, h) # screen x is calculated during render
        
        # Create a jagged polygon for visual flair
        points = []
        num_points = 8
        for i in range(num_points):
            angle = 2 * math.pi * i / num_points
            dist = self.np_random.uniform(0.8, 1.2)
            px = w/2 * dist * math.cos(angle) + w/2
            py = h/2 * dist * math.sin(angle) + h/2
            points.append((px, py))

        self.obstacles.append({'world_x': obs_world_x, 'rect': rect, 'poly_points': points})
        
        # Set next spawn distance
        min_dist = 250 - self.stage * 20
        max_dist = 500 - self.stage * 40
        self.next_obstacle_spawn_dist = self.camera_x + self.np_random.integers(min_dist, max_dist)

    def _spawn_bonus(self):
        bonus_world_x = self.camera_x + self.WIDTH + self.np_random.integers(500, 1000)
        y = self.np_random.integers(100, self.GROUND_Y - 100)
        size = 15
        rect = pygame.Rect(0, y, size, size)
        self.bonuses.append({'world_x': bonus_world_x, 'rect': rect})
        self.next_bonus_spawn_dist = self.camera_x + self.np_random.integers(1000, 2000)

    def _handle_collisions(self):
        reward = 0
        player_rect = pygame.Rect(
            self.player_pos[0] - self.PLAYER_RADIUS,
            self.player_pos[1] - self.PLAYER_RADIUS,
            self.PLAYER_RADIUS * 2,
            self.PLAYER_RADIUS * 2,
        )

        # Obstacles
        for obs in self.obstacles:
            obs_screen_x = obs['world_x'] - self.camera_x
            obs_rect = obs['rect'].copy()
            obs_rect.x = obs_screen_x
            if player_rect.colliderect(obs_rect):
                self.lives -= 1
                reward -= 10
                # sfx: explosion_sound()
                self._create_particles(self.player_pos, 50, self.COLOR_OBSTACLE, 'explosion')
                self.obstacles.remove(obs)
                if self.lives <= 0:
                    self.game_over = True
                break
        
        # Bonuses
        for bonus in self.bonuses:
            bonus_screen_x = bonus['world_x'] - self.camera_x
            bonus_rect = bonus['rect'].copy()
            bonus_rect.x = bonus_screen_x
            if player_rect.colliderect(bonus_rect):
                reward += 5
                # sfx: bonus_pickup_sound()
                self._create_particles((bonus_screen_x + bonus_rect.width/2, bonus_rect.centery), 30, self.COLOR_BONUS, 'collect')
                self.bonuses.remove(bonus)
                break
        
        return reward

    def _advance_stage(self):
        self.stage += 1
        if self.stage > self.MAX_STAGES:
            self.victory = True
            self.game_over = True
            self.score += 100 # Victory bonus
        else:
            # sfx: stage_clear_sound()
            self.scroll_speed += 0.5 # Corresponds to brief's 0.05 units/sec for 10 steps/sec
            self.stage_timer = 0

    def _check_termination(self):
        if self.game_over:
            return True
        if self.stage_timer >= self.TIME_PER_STAGE_STEPS:
            self.game_over = True
            # sfx: time_out_sound()
            return True
        return False

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_background()
        self._render_game_objects()
        self._render_particles()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_background(self):
        # Ground
        pygame.draw.rect(self.screen, self.COLOR_GROUND, (0, self.GROUND_Y, self.WIDTH, self.HEIGHT - self.GROUND_Y))
        
        # Parallax Stars
        for i, layer in enumerate(self.stars):
            scroll_factor = 0.1 + 0.2 * i
            for star in layer:
                x = (star[0] - self.camera_x * scroll_factor) % self.WIDTH
                pygame.gfxdraw.pixel(self.screen, int(x), int(star[1]), self.COLOR_STARS[i])

    def _render_game_objects(self):
        # Obstacles
        for obs in self.obstacles:
            screen_x = int(obs['world_x'] - self.camera_x)
            r = obs['rect']
            
            # Draw polygon relative to its top-left corner
            poly_points_on_screen = [(p[0] + screen_x, p[1] + r.y) for p in obs['poly_points']]
            if len(poly_points_on_screen) > 2:
                pygame.gfxdraw.aapolygon(self.screen, poly_points_on_screen, self.COLOR_OBSTACLE_OUTLINE)
                pygame.gfxdraw.filled_polygon(self.screen, poly_points_on_screen, self.COLOR_OBSTACLE)

        # Bonuses
        for bonus in self.bonuses:
            screen_x = int(bonus['world_x'] - self.camera_x)
            r = bonus['rect']
            pygame.draw.rect(self.screen, self.COLOR_BONUS, (screen_x, r.y, r.width, r.height), border_radius=4)
            
        # Player
        if self.lives > 0:
            px, py = int(self.player_pos[0]), int(self.player_pos[1])
            # Glow
            pygame.gfxdraw.filled_circle(self.screen, px, py, self.PLAYER_RADIUS + 4, (*self.COLOR_PLAYER_GLOW, 50))
            pygame.gfxdraw.filled_circle(self.screen, px, py, self.PLAYER_RADIUS + 8, (*self.COLOR_PLAYER_GLOW, 30))
            # Ship
            pygame.gfxdraw.aacircle(self.screen, px, py, self.PLAYER_RADIUS, self.COLOR_PLAYER)
            pygame.gfxdraw.filled_circle(self.screen, px, py, self.PLAYER_RADIUS, self.COLOR_PLAYER)

    def _render_ui(self):
        # Score
        score_text = self.font_ui.render(f"SCORE: {int(self.score):06d}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))

        # Lives
        life_text = self.font_ui.render("LIVES: ", True, self.COLOR_TEXT)
        self.screen.blit(life_text, (self.WIDTH - 180, 10))
        for i in range(self.lives):
            pygame.gfxdraw.filled_circle(self.screen, self.WIDTH - 90 + i * 25, 22, 8, self.COLOR_PLAYER)
            pygame.gfxdraw.aacircle(self.screen, self.WIDTH - 90 + i * 25, 22, 8, self.COLOR_PLAYER)

        # Stage and Time
        time_left = max(0, self.TIME_PER_STAGE_S - self.stage_timer // self.FPS)
        time_color = self.COLOR_TEXT if time_left > 10 else self.COLOR_TIMER_WARN
        stage_text = self.font_ui.render(f"STAGE: {self.stage}/{self.MAX_STAGES}   TIME: {time_left}", True, time_color)
        stage_rect = stage_text.get_rect(centerx=self.WIDTH // 2, y=10)
        self.screen.blit(stage_text, stage_rect)

        # Game Over / Victory Message
        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            self.screen.blit(overlay, (0, 0))
            
            if self.victory:
                msg = "VICTORY!"
            else:
                msg = "GAME OVER"
            
            end_text = self.font_game_over.render(msg, True, self.COLOR_TEXT)
            text_rect = end_text.get_rect(center=(self.WIDTH // 2, self.HEIGHT // 2))
            self.screen.blit(end_text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "lives": self.lives,
            "stage": self.stage,
            "distance": self.camera_x,
        }

    # --- Particle System ---
    def _create_particles(self, pos, count, color, p_type):
        for _ in range(count):
            if p_type == 'explosion':
                vel = np.array([self.np_random.uniform(-4, 4), self.np_random.uniform(-4, 4)])
                life = self.np_random.integers(15, 30)
            elif p_type == 'thrust':
                vel = np.array([self.np_random.uniform(-1, 1), self.np_random.uniform(1, 3)])
                life = self.np_random.integers(10, 20)
            elif p_type == 'land':
                vel = np.array([self.np_random.uniform(-2, 2), self.np_random.uniform(-2, 0)])
                life = self.np_random.integers(8, 15)
            elif p_type == 'collect':
                 vel = np.array([self.np_random.uniform(-3, 3), self.np_random.uniform(-3, 3)])
                 life = self.np_random.integers(10, 25)
            else:
                vel = np.array([0.0, 0.0])
                life = 10
            
            self.particles.append({'pos': np.array(pos), 'vel': vel, 'life': life, 'max_life': life, 'color': color})

    def _update_particles(self):
        for p in self.particles:
            p['pos'] += p['vel']
            p['life'] -= 1
        self.particles = [p for p in self.particles if p['life'] > 0]
    
    def _render_particles(self):
        for p in self.particles:
            life_ratio = p['life'] / p['max_life']
            size = int(life_ratio * 5)
            if size > 0:
                color = (*p['color'], int(life_ratio * 255))
                pygame.gfxdraw.filled_circle(self.screen, int(p['pos'][0]), int(p['pos'][1]), size, color)

    # --- Utility Functions ---
    def _generate_initial_stars(self):
        self.stars = []
        for _ in self.COLOR_STARS:
            layer = []
            for _ in range(150):
                layer.append((self.np_random.integers(0, self.WIDTH), self.np_random.integers(0, self.GROUND_Y)))
            self.stars.append(layer)

    def validate_implementation(self):
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        test_obs = self._get_observation()
        assert test_obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert test_obs.dtype == np.uint8
        obs, info = self.reset()
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(info, dict)
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert not trunc
        assert isinstance(info, dict)
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to run the file directly to play the game
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    running = True
    game_window = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Hopping Spaceship")
    clock = pygame.time.Clock()

    total_reward = 0
    
    while running:
        # --- Player Input ---
        keys = pygame.key.get_pressed()
        move = 0 # Unused in this game
        space = 1 if keys[pygame.K_SPACE] else 0
        shift = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        action = [move, space, shift]

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                print("Resetting environment")
                obs, info = env.reset()
                total_reward = 0

        # --- Environment Step ---
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if terminated or truncated:
            print(f"Episode finished. Total Reward: {total_reward:.2f}. Score: {info['score']:.2f}. Press 'R' to restart.")
            # In a real game, you might wait for a key press before resetting.
            # Here we just let it sit on the game over screen.

        # --- Rendering ---
        # The observation is already a rendered frame
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        game_window.blit(surf, (0, 0))
        pygame.display.flip()

        clock.tick(env.FPS)

    pygame.quit()