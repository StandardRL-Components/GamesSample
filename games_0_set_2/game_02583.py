import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
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
        "Controls: Press space to jump. Timing is everything."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A minimalist platformer. Time your jumps to ascend through three increasingly difficult stages before the timer runs out."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    FPS = 30
    NUM_STAGES = 3

    COLOR_BG = (20, 30, 40)
    COLOR_PLAYER = (255, 255, 255)
    COLOR_PLATFORM = (100, 110, 120)
    COLOR_GOAL = (0, 200, 120)
    COLOR_PARTICLE = (200, 200, 220)
    COLOR_TEXT = (230, 230, 230)
    COLOR_UI_BG = (30, 45, 60, 180) # RGBA for transparency

    PLAYER_SIZE = 20
    GRAVITY = 0.8
    JUMP_STRENGTH = -15.0

    STAGE_TIME_LIMIT_SECONDS = 60

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Gymnasium spaces
        self.observation_space = Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.SysFont("Consolas", 48, bold=True)
        self.font_medium = pygame.font.SysFont("Consolas", 24)
        self.font_small = pygame.font.SysFont("Consolas", 18)

        # Game state variables
        self.player_pos = None
        self.player_vel_y = None
        self.is_grounded = None
        self.prev_space_held = None
        self.player_rect = None

        self.platforms = None
        self.goal_platform_idx = None
        self.highest_platform_reached = None

        self.current_stage = None
        self.stage_timer = None
        
        self.camera_y = None
        self.particles = None

        self.steps = None
        self.score = None
        self.game_over = None
        self.win = None
        
    def _generate_platforms(self, stage_num):
        platforms = []
        plat_y = self.SCREEN_HEIGHT - 40
        
        # Starting platform
        start_plat = pygame.Rect(self.SCREEN_WIDTH // 2 - 75, plat_y, 150, 20)
        platforms.append(start_plat)
        
        # Stage-dependent parameters
        base_width = 120
        base_y_gap_min = 100
        base_y_gap_max = 150
        
        plat_width = int(base_width * (1 - 0.2 * (stage_num - 1)))
        y_gap_min = int(base_y_gap_min * (1 + 0.15 * (stage_num - 1)))
        y_gap_max = int(base_y_gap_max * (1 + 0.15 * (stage_num - 1)))

        num_platforms = 15
        for i in range(num_platforms):
            plat_y -= self.np_random.integers(y_gap_min, y_gap_max + 1)
            plat_x = self.np_random.integers(
                0, self.SCREEN_WIDTH - plat_width
            )
            platforms.append(pygame.Rect(plat_x, plat_y, plat_width, 20))
            
        # Goal platform
        plat_y -= self.np_random.integers(y_gap_min, y_gap_max + 1)
        goal_width = 80
        plat_x = self.np_random.integers(0, self.SCREEN_WIDTH - goal_width)
        platforms.append(pygame.Rect(plat_x, plat_y, goal_width, 20))
        
        return platforms

    def _setup_stage(self, stage_num):
        self.current_stage = stage_num
        self.stage_timer = self.STAGE_TIME_LIMIT_SECONDS * self.FPS
        self.platforms = self._generate_platforms(stage_num)
        self.goal_platform_idx = len(self.platforms) - 1
        
        start_platform = self.platforms[0]
        self.player_pos = [start_platform.centerx, start_platform.top - self.PLAYER_SIZE]
        self.player_rect = pygame.Rect(self.player_pos[0], self.player_pos[1], self.PLAYER_SIZE, self.PLAYER_SIZE)
        self.player_vel_y = 0
        self.is_grounded = True
        self.prev_space_held = True # Prevent jumping on first frame
        self.highest_platform_reached = 0
        
        self.particles = []
        self.camera_y = self.player_pos[1] - self.SCREEN_HEIGHT * 2 / 3
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False
        
        self._setup_stage(1)
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        reward = 0
        terminated = False
        
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        # --- 1. Handle Input ---
        space_held = action[1] == 1
        jump_action = space_held and not self.prev_space_held

        if jump_action and self.is_grounded:
            self.player_vel_y = self.JUMP_STRENGTH
            self.is_grounded = False
        
        self.prev_space_held = space_held

        # --- 2. Update Physics ---
        # Apply gravity
        self.player_vel_y += self.GRAVITY
        self.player_vel_y = min(self.player_vel_y, 15) # Terminal velocity

        # Update position
        player_prev_rect = pygame.Rect(self.player_pos[0], self.player_pos[1], self.PLAYER_SIZE, self.PLAYER_SIZE)
        self.player_pos[1] += self.player_vel_y
        self.player_rect.topleft = self.player_pos

        # Collision detection
        landed_on_platform = False
        if self.player_vel_y > 0: # Only check for landing if falling
            for i, plat in enumerate(self.platforms):
                if self.player_rect.colliderect(plat) and player_prev_rect.bottom <= plat.top:
                    self.player_pos[1] = plat.top - self.PLAYER_SIZE
                    self.player_rect.topleft = self.player_pos
                    self.player_vel_y = 0
                    self.is_grounded = True
                    landed_on_platform = True
                    
                    self._create_particles(self.player_rect.midbottom)

                    if i > self.highest_platform_reached:
                        reward += 10 # Reward for reaching a new, higher platform
                        self.highest_platform_reached = i
                    
                    # Check for goal collision
                    if i == self.goal_platform_idx:
                        reward += 50 # Stage completion reward
                        if self.current_stage < self.NUM_STAGES:
                            self._setup_stage(self.current_stage + 1)
                        else:
                            # Game won
                            self.win = True
                            self.game_over = True
                            terminated = True
                            reward += 100 # Game completion bonus
                    break
        
        if not landed_on_platform:
            self.is_grounded = False
        
        if self.is_grounded:
            reward += 0.1 # Small reward for being on a platform

        # --- 3. Update Game State ---
        self.steps += 1
        self.stage_timer -= 1
        
        # Update particles
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['life'] -= 1
        self.particles = [p for p in self.particles if p['life'] > 0]
        
        # --- 4. Check Termination Conditions ---
        # Fell off screen
        if self.player_pos[1] > self.camera_y + self.SCREEN_HEIGHT:
            self.game_over = True
            terminated = True
            reward = -100 # Penalty for falling
        
        # Ran out of time
        if self.stage_timer <= 0 and not self.game_over:
            self.game_over = True
            terminated = True
            reward = -50 # Penalty for timeout
        
        self.score += reward
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _create_particles(self, position):
        for _ in range(10):
            angle = self.np_random.uniform(math.pi, 2 * math.pi)
            speed = self.np_random.uniform(0.5, 2.5)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            life = self.np_random.integers(15, 30)
            self.particles.append({'pos': list(position), 'vel': vel, 'life': life})

    def _get_observation(self):
        # Update camera
        target_camera_y = self.player_pos[1] - self.SCREEN_HEIGHT * 2 / 3
        self.camera_y += (target_camera_y - self.camera_y) * 0.1

        # Clear screen
        self.screen.fill(self.COLOR_BG)
        
        # Render game elements
        self._render_game()
        
        # Render UI
        self._render_ui()
        
        # Convert to numpy array
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Render platforms
        for i, plat in enumerate(self.platforms):
            color = self.COLOR_GOAL if i == self.goal_platform_idx else self.COLOR_PLATFORM
            render_rect = plat.move(0, -self.camera_y)
            pygame.draw.rect(self.screen, color, render_rect, border_radius=3)
            
        # Render particles
        for p in self.particles:
            pos = (int(p['pos'][0]), int(p['pos'][1] - self.camera_y))
            radius = int(p['life'] / 5)
            if radius > 0:
                pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], radius, self.COLOR_PARTICLE)
        
        # Render player
        player_render_rect = self.player_rect.move(0, -self.camera_y)
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, player_render_rect)
        # Add a small "glow" for visibility
        glow_rect = player_render_rect.inflate(4, 4)
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, glow_rect, 1, border_radius=2)

    def _render_ui(self):
        # Stage text
        stage_text = self.font_medium.render(f"Stage: {self.current_stage}/{self.NUM_STAGES}", True, self.COLOR_TEXT)
        self.screen.blit(stage_text, (10, 10))
        
        # Score text
        score_text = self.font_medium.render(f"Score: {int(self.score)}", True, self.COLOR_TEXT)
        score_rect = score_text.get_rect(centerx=self.SCREEN_WIDTH / 2, y=10)
        self.screen.blit(score_text, score_rect)

        # Timer text
        time_left = max(0, self.stage_timer / self.FPS)
        timer_color = (255, 100, 100) if time_left < 10 else self.COLOR_TEXT
        timer_text = self.font_medium.render(f"Time: {time_left:.1f}", True, timer_color)
        timer_rect = timer_text.get_rect(right=self.SCREEN_WIDTH - 10, y=10)
        self.screen.blit(timer_text, timer_rect)
        
        # Game Over / Win message
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill(self.COLOR_UI_BG)
            self.screen.blit(overlay, (0, 0))
            
            message = "YOU WIN!" if self.win else "GAME OVER"
            end_text = self.font_large.render(message, True, self.COLOR_TEXT)
            end_rect = end_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(end_text, end_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "stage": self.current_stage,
            "time_left": max(0, self.stage_timer / self.FPS),
        }
        
    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # This block allows you to play the game directly
    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    # --- Manual Control Setup ---
    # Action map: 0=none, 1=up, 2=down, 3=left, 4=right
    #             0=space_up, 1=space_down
    #             0=shift_up, 1=shift_down
    action = [0, 0, 0] 
    total_reward = 0
    
    # Create a window to display the game
    real_screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Minimalist Platformer")
    
    # Game loop
    while not done:
        # --- Human Input ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    action[1] = 1
                if event.key == pygame.K_r: # Reset on 'r'
                    obs, info = env.reset()
                    total_reward = 0
            if event.type == pygame.KEYUP:
                if event.key == pygame.K_SPACE:
                    action[1] = 0

        # --- Environment Step ---
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        done = terminated or truncated

        # --- Rendering ---
        # The observation is already a rendered frame
        # We just need to get it to the display screen
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        real_screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        # Control frame rate
        env.clock.tick(GameEnv.FPS)

    print(f"Game Over! Final Score: {total_reward}")
    env.close()