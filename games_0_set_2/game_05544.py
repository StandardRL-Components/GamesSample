
# Generated: 2025-08-28T05:19:47.465143
# Source Brief: brief_05544.md
# Brief Index: 5544

        
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
        "Controls: ←→ to run. Press ↑ or Space to jump. Avoid obstacles and reach the exit!"
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Escape a haunted mansion by running and jumping over cursed furniture and ghosts. "
        "You have 3 minutes to clear 3 stages. Touching 7 obstacles means you're trapped forever!"
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    FLOOR_Y = 350
    FPS = 30

    # Colors
    COLOR_BG = (15, 10, 25)
    COLOR_WALL = (25, 20, 40)
    COLOR_FLOOR = (40, 35, 55)
    COLOR_PLAYER = (255, 50, 50)
    COLOR_PLAYER_GLOW = (255, 100, 100, 50)
    COLOR_GHOST = (200, 220, 255)
    COLOR_FURNITURE = (80, 60, 40)
    COLOR_EXIT = (255, 223, 0)
    COLOR_EXIT_GLOW = (255, 223, 0, 80)
    COLOR_TEXT = (240, 240, 240)
    COLOR_TIMER_WARN = (255, 100, 100)
    COLOR_PARTICLE_HIT = (255, 150, 150)
    COLOR_PARTICLE_JUMP = (200, 200, 200)

    # Physics
    GRAVITY = 0.8
    PLAYER_JUMP_STRENGTH = -14
    PLAYER_SPEED = 6.0
    PLAYER_WIDTH = 20
    PLAYER_HEIGHT = 40

    # Game rules
    MAX_OBSTACLE_HITS = 7
    TOTAL_TIME_SECONDS = 180
    TOTAL_STAGES = 3
    STAGE_LENGTH_PIXELS = 4000

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
        self.font_large = pygame.font.SysFont("monospace", 24, bold=True)
        self.font_small = pygame.font.SysFont("monospace", 16)
        
        self.game_over_font = pygame.font.SysFont("monospace", 50, bold=True)

        self.reset()
        
        # This can be uncommented for self-validation during development
        # self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False

        self.player_pos = np.array([150.0, self.FLOOR_Y - self.PLAYER_HEIGHT])
        self.player_vel = np.array([0.0, 0.0])
        self.on_ground = True
        
        self.camera_x = 0.0

        self.obstacle_hits = 0
        self.time_remaining_frames = self.TOTAL_TIME_SECONDS * self.FPS
        self.current_stage = 1

        self.obstacles = []
        self.particles = []
        self.torches = []
        self.background_elements = []

        self.no_jump_counter = 0
        self.last_reward = 0.0

        self._generate_stage()

        return self._get_observation(), self._get_info()

    def _generate_stage(self):
        self.obstacles.clear()
        self.player_pos = np.array([150.0, self.FLOOR_Y - self.PLAYER_HEIGHT])
        self.camera_x = 0.0

        # Place exit door
        self.exit_pos = np.array([float(self.STAGE_LENGTH_PIXELS), float(self.FLOOR_Y - 80)])
        
        # Generate torches and background elements for parallax
        self.torches.clear()
        self.background_elements.clear()
        for i in range(20):
            x = self.np_random.uniform(0, self.STAGE_LENGTH_PIXELS)
            y = self.np_random.uniform(100, self.FLOOR_Y - 50)
            self.torches.append({'pos': [x, y], 'flicker': self.np_random.uniform(0.5, 1.5)})
        for i in range(50):
            x = self.np_random.uniform(0, self.STAGE_LENGTH_PIXELS)
            y = self.np_random.uniform(50, self.FLOOR_Y - 20)
            size = self.np_random.uniform(5, 20)
            self.background_elements.append({'rect': pygame.Rect(x, y, size, size*2), 'depth': self.np_random.uniform(0.7, 0.9)})

    def _spawn_obstacle(self):
        # Increase spawn rate and difficulty with stages
        spawn_chance = 0.015 + self.current_stage * 0.005
        if self.np_random.random() < spawn_chance:
            obstacle_type = self.np_random.choice(['furniture', 'ghost'])
            x_pos = self.camera_x + self.SCREEN_WIDTH + self.np_random.uniform(50, 150)
            
            base_speed = 1.0 + (self.current_stage - 1) * 0.5 # Brief specified 0.05/frame, this is per-stage increase
            
            if obstacle_type == 'furniture':
                width = self.np_random.uniform(30, 60)
                height = self.np_random.uniform(30, 80)
                rect = pygame.Rect(x_pos, self.FLOOR_Y - height, width, height)
                self.obstacles.append({'rect': rect, 'type': 'furniture'})
            elif obstacle_type == 'ghost':
                size = self.np_random.uniform(30, 50)
                y_pos = self.FLOOR_Y - size - self.np_random.uniform(20, 120)
                rect = pygame.Rect(x_pos, y_pos, size, size)
                self.obstacles.append({
                    'rect': rect, 'type': 'ghost', 'initial_y': y_pos,
                    'amplitude': self.np_random.uniform(20, 60),
                    'frequency': self.np_random.uniform(0.02, 0.05),
                    'speed': base_speed + self.np_random.uniform(-0.5, 0.5)
                })

    def _create_particles(self, pos, count, color, speed_range, life_range):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(*speed_range)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            life = self.np_random.integers(*life_range)
            self.particles.append({'pos': list(pos), 'vel': vel, 'life': life, 'max_life': life, 'color': color})

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        reward = 0.0

        # --- Handle Player Input ---
        is_jumping = (movement == 1 or space_held)
        if is_jumping and self.on_ground:
            self.player_vel[1] = self.PLAYER_JUMP_STRENGTH
            self.on_ground = False
            self.no_jump_counter = 0
            # sfx: jump.wav
            self._create_particles(self.player_pos + [self.PLAYER_WIDTH/2, self.PLAYER_HEIGHT], 10, self.COLOR_PARTICLE_JUMP, (1, 3), (10, 20))
        
        if not is_jumping:
            self.no_jump_counter += 1
        
        if movement == 3:  # Left
            self.player_vel[0] = -self.PLAYER_SPEED
        elif movement == 4:  # Right
            self.player_vel[0] = self.PLAYER_SPEED
        else:
            self.player_vel[0] = 0

        # --- Update Game State ---
        self.steps += 1
        self.time_remaining_frames -= 1
        
        # Update player physics
        self.player_vel[1] += self.GRAVITY
        self.player_pos += self.player_vel
        
        # Keep player on screen (horizontally)
        self.player_pos[0] = np.clip(self.player_pos[0], self.camera_x, self.camera_x + self.SCREEN_WIDTH - self.PLAYER_WIDTH)
        
        if self.player_pos[1] + self.PLAYER_HEIGHT >= self.FLOOR_Y:
            if not self.on_ground:
                # sfx: land.wav
                self._create_particles(self.player_pos + [self.PLAYER_WIDTH/2, self.PLAYER_HEIGHT], 5, self.COLOR_PARTICLE_JUMP, (0.5, 2), (5, 15))
            self.player_pos[1] = self.FLOOR_Y - self.PLAYER_HEIGHT
            self.player_vel[1] = 0
            self.on_ground = True

        # Update Camera
        target_camera_x = self.player_pos[0] - self.SCREEN_WIDTH * 0.25
        self.camera_x += (target_camera_x - self.camera_x) * 0.1
        self.camera_x = max(0, self.camera_x)

        # Update obstacles
        player_rect = pygame.Rect(self.player_pos[0], self.player_pos[1], self.PLAYER_WIDTH, self.PLAYER_HEIGHT)
        risky_jump_reward = 0
        
        for obs in self.obstacles[:]:
            if obs['type'] == 'ghost':
                obs['rect'].x -= obs['speed']
                obs['rect'].y = obs['initial_y'] + math.sin(self.steps * obs['frequency']) * obs['amplitude']
            
            # Risky jump reward check
            dist_to_obs = math.hypot(player_rect.centerx - obs['rect'].centerx, player_rect.centery - obs['rect'].centery)
            if dist_to_obs < 100 and not self.on_ground:
                risky_jump_reward = max(risky_jump_reward, 0.5)

            if player_rect.colliderect(obs['rect']):
                self.obstacle_hits += 1
                self.obstacles.remove(obs)
                self.score -= 50
                # sfx: hit_obstacle.wav
                self._create_particles(player_rect.center, 20, self.COLOR_PARTICLE_HIT, (2, 5), (20, 40))
                # Terminal reward handled below
            elif obs['rect'].right < self.camera_x:
                self.obstacles.remove(obs)
        
        # Update particles
        for p in self.particles[:]:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['life'] -= 1
            if p['life'] <= 0:
                self.particles.remove(p)

        # Spawn new obstacles
        self._spawn_obstacle()

        # --- Calculate Reward ---
        reward += 0.1  # Survival reward
        reward += risky_jump_reward # Risky jump reward
        
        if self.no_jump_counter > 30: # Longer than 1 sec
            reward -= 0.2 # Penalty for inactivity
            self.no_jump_counter = 0 # Reset to avoid constant penalty

        # --- Check for Stage Clear ---
        if player_rect.right > self.exit_pos[0]:
            self.score += 1000 * self.current_stage
            reward += 10.0
            self.current_stage += 1
            if self.current_stage > self.TOTAL_STAGES:
                self.win = True
                self.game_over = True
                reward += 100.0  # Big win bonus
                # sfx: game_win.wav
            else:
                self._generate_stage()
                # sfx: stage_clear.wav

        # --- Check Termination ---
        terminated = False
        if self.obstacle_hits >= self.MAX_OBSTACLE_HITS or self.time_remaining_frames <= 0:
            self.game_over = True
        
        if self.game_over:
            terminated = True
            if not self.win:
                reward -= 100.0 # Big loss penalty
                # sfx: game_over.wav
        
        self.score += reward
        self.last_reward = reward

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _get_observation(self):
        # --- Rendering ---
        # Background
        self.screen.fill(self.COLOR_BG)
        
        # Parallax background elements
        for elem in self.background_elements:
            screen_x = elem['rect'].x - self.camera_x * elem['depth']
            if -elem['rect'].width < screen_x < self.SCREEN_WIDTH:
                 pygame.draw.rect(self.screen, self.COLOR_WALL, (screen_x, elem['rect'].y, elem['rect'].width, elem['rect'].height))
        
        # Floor
        pygame.draw.rect(self.screen, self.COLOR_FLOOR, (0, self.FLOOR_Y, self.SCREEN_WIDTH, self.SCREEN_HEIGHT - self.FLOOR_Y))

        # Torches
        for torch in self.torches:
            screen_x = torch['pos'][0] - self.camera_x
            if -20 < screen_x < self.SCREEN_WIDTH + 20:
                flicker_size = 10 + math.sin(self.steps * 0.1 * torch['flicker']) * 3
                flicker_color = (255, self.np_random.integers(150, 200), 0)
                pygame.gfxdraw.filled_circle(self.screen, int(screen_x), int(torch['pos'][1]), int(flicker_size) + 5, (100, 80, 0, 50))
                pygame.gfxdraw.filled_circle(self.screen, int(screen_x), int(torch['pos'][1]), int(flicker_size), flicker_color)

        # Exit door
        exit_screen_x = self.exit_pos[0] - self.camera_x
        exit_rect = pygame.Rect(exit_screen_x, self.exit_pos[1], 50, 80)
        pygame.gfxdraw.box(self.screen, exit_rect.inflate(20, 20), self.COLOR_EXIT_GLOW)
        pygame.draw.rect(self.screen, self.COLOR_EXIT, exit_rect)
        pygame.draw.rect(self.screen, (0,0,0), exit_rect.inflate(-10, -10))

        # Obstacles
        for obs in self.obstacles:
            obs_screen_rect = obs['rect'].move(-self.camera_x, 0)
            if obs['type'] == 'furniture':
                pygame.draw.rect(self.screen, self.COLOR_FURNITURE, obs_screen_rect)
                pygame.draw.rect(self.screen, tuple(np.clip(np.array(self.COLOR_FURNITURE)*1.2, 0, 255)), obs_screen_rect, 2)
            elif obs['type'] == 'ghost':
                surf = pygame.Surface(obs_screen_rect.size, pygame.SRCALPHA)
                pygame.draw.ellipse(surf, (*self.COLOR_GHOST, 128), (0, 0, *obs_screen_rect.size))
                self.screen.blit(surf, obs_screen_rect.topleft)

        # Particles
        for p in self.particles:
            alpha = int(255 * (p['life'] / p['max_life']))
            color = (*p['color'], alpha)
            pos = (p['pos'][0] - self.camera_x, p['pos'][1])
            pygame.draw.circle(self.screen, color, pos, int(p['life'] * 0.2 + 1))

        # Player
        player_screen_pos = self.player_pos - [self.camera_x, 0]
        player_rect = pygame.Rect(player_screen_pos, (self.PLAYER_WIDTH, self.PLAYER_HEIGHT))
        
        # Player glow
        pygame.gfxdraw.box(self.screen, player_rect.inflate(15, 15), self.COLOR_PLAYER_GLOW)
        
        # Player body with simple animation
        anim_offset = 0
        if not self.on_ground:
            anim_offset = -3 # Stretch when jumping
        else:
            anim_offset = math.sin(self.steps * 0.5) * 2 # Bob when on ground
        
        animated_rect = pygame.Rect(player_rect.x, player_rect.y + anim_offset, player_rect.width, player_rect.height - anim_offset)
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, animated_rect, border_radius=3)
        
        # --- UI Overlay ---
        self._render_ui()

        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_ui(self):
        # Score
        score_text = self.font_large.render(f"SCORE: {int(self.score)}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))
        
        # Timer
        time_s = max(0, self.time_remaining_frames / self.FPS)
        mins, secs = divmod(time_s, 60)
        timer_color = self.COLOR_TEXT if time_s > 10 else self.COLOR_TIMER_WARN
        timer_text = self.font_large.render(f"TIME: {int(mins):02}:{int(secs):02}", True, timer_color)
        self.screen.blit(timer_text, (self.SCREEN_WIDTH - timer_text.get_width() - 10, 10))

        # Obstacle Hits
        hits_text = self.font_small.render(f"HITS: {self.obstacle_hits}/{self.MAX_OBSTACLE_HITS}", True, self.COLOR_TEXT)
        self.screen.blit(hits_text, (10, self.SCREEN_HEIGHT - hits_text.get_height() - 10))

        # Stage
        stage_text = self.font_small.render(f"STAGE: {self.current_stage}/{self.TOTAL_STAGES}", True, self.COLOR_TEXT)
        self.screen.blit(stage_text, (self.SCREEN_WIDTH - stage_text.get_width() - 10, self.SCREEN_HEIGHT - stage_text.get_height() - 10))

        # Game Over / Win message
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            self.screen.blit(overlay, (0, 0))
            
            message = "YOU ESCAPED!" if self.win else "GAME OVER"
            color = self.COLOR_EXIT if self.win else self.COLOR_TIMER_WARN
            
            end_text = self.game_over_font.render(message, True, color)
            text_rect = end_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(end_text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "stage": self.current_stage,
            "obstacle_hits": self.obstacle_hits,
            "time_remaining_seconds": self.time_remaining_frames / self.FPS,
        }

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation.
        '''
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

# Example of how to run the environment
if __name__ == '__main__':
    env = GameEnv(render_mode="rgb_array")
    
    # --- For manual play ---
    # This requires setting up a pygame display window
    
    # Set render_mode to "human" in __init__ if you add that mode.
    # For now, we'll just show the rgb_array in a window.
    
    pygame.display.set_caption("Haunted House Escape")
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    
    obs, info = env.reset()
    terminated = False
    
    # Game loop
    running = True
    while running:
        action = [0, 0, 0] # Default no-op
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            action[0] = 3
        elif keys[pygame.K_RIGHT]:
            action[0] = 4
            
        if keys[pygame.K_UP]:
            action[0] = 1
            
        if keys[pygame.K_SPACE]:
            action[1] = 1
            
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            action[2] = 1

        if terminated:
            # Simple reset after 3 seconds on game over screen
            pygame.time.wait(3000)
            obs, info = env.reset()
            terminated = False

        obs, reward, terminated, truncated, info = env.step(action)

        # Display the observation
        # Pygame uses (width, height), numpy uses (height, width)
        # The observation is (H, W, C), so we need to transpose it for pygame's surfarray
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        env.clock.tick(env.FPS)

    env.close()