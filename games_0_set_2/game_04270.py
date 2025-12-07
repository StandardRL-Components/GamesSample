import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import os
import pygame



class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: ↑↓ to aim jump, ←→ to change power. Space to jump. Reach the gold platform at the top!"
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Leap between procedurally generated platforms to reach the top in this fast-paced, side-scrolling arcade hopper."
    )

    # Frames auto-advance for real-time gameplay.
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Game constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 30
        self.MAX_STAGES = 3
        self.STAGE_TIME_LIMIT = 60  # seconds
        self.MAX_STEPS = self.FPS * self.STAGE_TIME_LIMIT * self.MAX_STAGES

        # Colors
        self.COLOR_BG_TOP = (10, 20, 40)
        self.COLOR_BG_BOTTOM = (0, 0, 10)
        self.COLOR_PLAYER = (255, 255, 255)
        self.COLOR_PLAYER_GLOW = (200, 200, 255)
        self.COLOR_TEXT = (255, 255, 255)
        self.COLOR_HEART = (255, 50, 50)
        self.COLOR_STAGE_TEXT = (255, 220, 0)
        self.PLATFORM_PALETTES = [
            [(50, 255, 150), (100, 255, 200), (0, 200, 100)], # Stage 1: Greens
            [(150, 50, 255), (200, 100, 255), (100, 0, 200)], # Stage 2: Purples
            [(255, 150, 50), (255, 200, 100), (200, 100, 0)], # Stage 3: Oranges
        ]
        self.COLOR_GOAL = (255, 215, 0)

        # Physics
        self.GRAVITY = 0.4
        self.MAX_JUMP_SPEED = 12
        self.PLAYER_SIZE = 12

        # Gymnasium spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        # Headless rendering
        os.environ["SDL_VIDEODRIVER"] = "dummy"
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("monospace", 20, bold=True)
        self.font_stage = pygame.font.SysFont("monospace", 24, bold=True)
        
        # Initialize state variables
        self.player_pos = None
        self.player_vel = None
        self.on_platform = None
        self.platforms = None
        self.particles = None
        self.jump_angle = None
        self.jump_power = None
        self.last_space_held = None
        self.camera_x = None
        self.steps = None
        self.score = None
        self.game_over = None
        self.game_won = None
        self.lives = None
        self.stage = None
        self.stage_timer = None
        self.max_height_reached = None
        self.current_platform_palette = None
        
        # self.reset() is called by the wrapper, but we can call it to init state
        # self.validate_implementation() is a helper and not part of the final env
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.game_won = False
        self.lives = 3
        self.stage = 1
        
        self._setup_stage()
        
        return self._get_observation(), self._get_info()

    def _setup_stage(self):
        """Initializes state for the current stage."""
        self.stage_timer = self.FPS * self.STAGE_TIME_LIMIT
        self.platforms = self._generate_platforms()
        start_platform = self.platforms[0]
        self.player_pos = pygame.math.Vector2(start_platform.centerx, start_platform.top - self.PLAYER_SIZE / 2)
        self.player_vel = pygame.math.Vector2(0, 0)
        self.on_platform = True
        self.particles = []
        self.jump_angle = 60  # Start with a decent angle
        self.jump_power = 0.75 # Start with decent power
        self.last_space_held = False
        self.camera_x = self.player_pos.x - self.WIDTH / 2
        self.max_height_reached = self.player_pos.y
        self.current_platform_palette = self.PLATFORM_PALETTES[(self.stage - 1) % len(self.PLATFORM_PALETTES)]

    def _generate_platforms(self):
        platforms = []
        
        # Starting platform
        base_width = 150
        start_plat = pygame.Rect(self.WIDTH // 2 - base_width // 2, self.HEIGHT - 40, base_width, 20)
        platforms.append(start_plat)

        # Procedural platforms
        num_platforms = 25
        last_plat = start_plat
        
        difficulty_mult = 1.0 - 0.05 * (self.stage - 1)
        
        for i in range(num_platforms):
            width = max(40, (random.uniform(60, 120) * difficulty_mult))
            height = 20
            
            # Ensure reachability
            angle = math.radians(random.uniform(30, 150))
            distance = random.uniform(80, 200) * (1 / difficulty_mult)
            
            dx = math.cos(angle) * distance
            dy = -math.sin(angle) * distance
            
            new_x = last_plat.centerx + dx
            new_y = last_plat.centery + dy
            
            # Clamp to reasonable world bounds
            new_x = np.clip(new_x, last_plat.centerx - 250, last_plat.centerx + 250)
            
            plat = pygame.Rect(int(new_x - width / 2), int(new_y - height / 2), int(width), int(height))
            platforms.append(plat)
            last_plat = plat
        
        # Goal platform
        goal_plat = platforms[-1]
        goal_plat.width = 150
        goal_plat.x -= (150 - goal_plat.width) / 2
        
        return platforms
    
    def step(self, action):
        reward = 0
        terminated = False
        truncated = False

        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # Unpack factorized action
        movement, space_pressed, _ = action[0], action[1] == 1, action[2] == 1
        
        # -- Update game logic --
        self.steps += 1
        self.stage_timer -= 1
        
        # Handle player controls if on a platform
        if self.on_platform:
            # Angle adjustment
            if movement == 1: self.jump_angle += 2
            elif movement == 2: self.jump_angle -= 2
            # Power adjustment
            elif movement == 4: self.jump_power += 0.02
            elif movement == 3: self.jump_power -= 0.02
            # Auto-adjust angle towards neutral
            elif movement == 0:
                if self.jump_angle > 45: self.jump_angle -= 0.5
                if self.jump_angle < 45: self.jump_angle += 0.5
            
            self.jump_angle = np.clip(self.jump_angle, 15, 90)
            self.jump_power = np.clip(self.jump_power, 0.5, 1.0)
            
            # Jump action (on rising edge of space press)
            if space_pressed and not self.last_space_held:
                self._jump()
        
        self.last_space_held = space_pressed
        
        # Update player physics if in the air
        if not self.on_platform:
            reward -= 0.01 # Penalty for being in the air
            self.player_vel.y += self.GRAVITY
            self.player_pos += self.player_vel
        
        # Collision detection and handling
        landed_platform = self._handle_collisions()
        if landed_platform:
            reward += 0.1 # Reward for landing
            
            # Check for stage completion
            if landed_platform == self.platforms[-1]:
                reward += 10.0
                self.score += 1000
                self.stage += 1
                if self.stage > self.MAX_STAGES:
                    self.game_won = True
                else:
                    self._setup_stage()
            
        # Update max height reward
        if self.player_pos.y < self.max_height_reached:
            reward += (self.max_height_reached - self.player_pos.y) * 0.1
            self.score += int(self.max_height_reached - self.player_pos.y)
            self.max_height_reached = self.player_pos.y

        # Handle failure conditions
        if self.player_pos.y > self.HEIGHT + 50 or self.stage_timer <= 0:
            reward -= 5.0
            self.lives -= 1
            if self.lives > 0:
                self._setup_stage() # Reset the stage
            else:
                self.game_over = True
        
        # Update camera
        self._update_camera()

        # Update particles
        self._update_particles()
        
        # -- Check termination --
        if self.game_won:
            reward += 100.0
            terminated = True
        elif self.game_over:
            terminated = True
        
        if self.steps >= self.MAX_STEPS:
            truncated = True

        self.score += reward

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _jump(self):
        self.on_platform = False
        angle_rad = math.radians(self.jump_angle)
        total_vel = self.MAX_JUMP_SPEED * self.jump_power
        self.player_vel.x = total_vel * math.cos(angle_rad)
        self.player_vel.y = -total_vel * math.sin(angle_rad)
    
    def _handle_collisions(self):
        if self.player_vel.y < 0: return None # Can only land when moving down

        player_rect = pygame.Rect(
            int(self.player_pos.x - self.PLAYER_SIZE // 2), 
            int(self.player_pos.y - self.PLAYER_SIZE // 2), 
            self.PLAYER_SIZE, self.PLAYER_SIZE
        )
        
        for i, plat in enumerate(self.platforms):
            if player_rect.colliderect(plat):
                # Check if player's bottom is just above the platform's top
                if abs(player_rect.bottom - plat.top) < 10 and self.player_pos.x > plat.left and self.player_pos.x < plat.right:
                    self.on_platform = True
                    self.player_pos.y = plat.top - self.PLAYER_SIZE / 2
                    self.player_vel = pygame.math.Vector2(0, 0)
                    self._create_particles(self.player_pos.x, plat.top)
                    return plat
        return None

    def _create_particles(self, x, y, count=10):
        for _ in range(count):
            self.particles.append({
                'pos': pygame.math.Vector2(x, y),
                'vel': pygame.math.Vector2(random.uniform(-1, 1), random.uniform(-1, 1)),
                'life': random.randint(10, 20),
                'radius': random.uniform(2, 4)
            })

    def _update_particles(self):
        for p in self.particles:
            p['pos'] += p['vel']
            p['life'] -= 1
        self.particles = [p for p in self.particles if p['life'] > 0]
    
    def _update_camera(self):
        # Smoothly follow the player with a look-ahead
        target_camera_x = self.player_pos.x - self.WIDTH / 2 + self.player_vel.x * 10
        self.camera_x = self.camera_x * 0.9 + target_camera_x * 0.1

    def _get_observation(self):
        self._render_background()
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_background(self):
        for y in range(self.HEIGHT):
            interp = y / self.HEIGHT
            color = (
                int(self.COLOR_BG_TOP[0] * (1 - interp) + self.COLOR_BG_BOTTOM[0] * interp),
                int(self.COLOR_BG_TOP[1] * (1 - interp) + self.COLOR_BG_BOTTOM[1] * interp),
                int(self.COLOR_BG_TOP[2] * (1 - interp) + self.COLOR_BG_BOTTOM[2] * interp),
            )
            pygame.draw.line(self.screen, color, (0, y), (self.WIDTH, y))

    def _render_game(self):
        # Render platforms
        for i, plat in enumerate(self.platforms):
            screen_pos = plat.copy()
            screen_pos.x -= self.camera_x
            color = self.COLOR_GOAL if i == len(self.platforms) - 1 else self.current_platform_palette[i % len(self.current_platform_palette)]
            pygame.draw.rect(self.screen, color, screen_pos, border_radius=3)
            pygame.draw.rect(self.screen, tuple(int(c*0.7) for c in color), screen_pos, width=2, border_radius=3)

        # Render jump trajectory indicator
        if self.on_platform:
            angle_rad = math.radians(self.jump_angle)
            line_len = 30 + 40 * self.jump_power
            end_x = self.player_pos.x + math.cos(angle_rad) * line_len - self.camera_x
            end_y = self.player_pos.y - math.sin(angle_rad) * line_len
            start_x = self.player_pos.x - self.camera_x
            start_y = self.player_pos.y
            pygame.draw.line(self.screen, self.COLOR_PLAYER, (start_x, start_y), (end_x, end_y), 2)

        # Render player
        player_screen_x = int(self.player_pos.x - self.camera_x)
        player_screen_y = int(self.player_pos.y)
        
        # Glow effect
        glow_radius = int(self.PLAYER_SIZE * 1.5)
        glow_surf = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
        pygame.draw.circle(glow_surf, (*self.COLOR_PLAYER_GLOW, 50), (glow_radius, glow_radius), glow_radius)
        self.screen.blit(glow_surf, (player_screen_x - glow_radius, player_screen_y - glow_radius - self.PLAYER_SIZE // 2), special_flags=pygame.BLEND_RGBA_ADD)

        player_rect = pygame.Rect(player_screen_x - self.PLAYER_SIZE//2, player_screen_y - self.PLAYER_SIZE, self.PLAYER_SIZE, self.PLAYER_SIZE)
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, player_rect, border_radius=2)
        
        # Render particles
        for p in self.particles:
            alpha = int(255 * (p['life'] / 20))
            color = (*self.COLOR_PLAYER, alpha)
            pos = (int(p['pos'].x - self.camera_x), int(p['pos'].y))
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], int(p['radius']), color)
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], int(p['radius']), color)

    def _render_ui(self):
        # Render lives
        for i in range(self.lives):
            self._draw_heart(25 + i * 35, 25)

        # Render time
        time_text = f"TIME: {max(0, self.stage_timer // self.FPS):02}"
        time_surf = self.font_ui.render(time_text, True, self.COLOR_TEXT)
        self.screen.blit(time_surf, (self.WIDTH - time_surf.get_width() - 15, 15))

        # Render stage
        stage_text = f"STAGE {self.stage}"
        stage_surf = self.font_stage.render(stage_text, True, self.COLOR_STAGE_TEXT)
        self.screen.blit(stage_surf, (self.WIDTH//2 - stage_surf.get_width()//2, 15))

        # Render score
        score_text = f"SCORE: {int(self.score)}"
        score_surf = self.font_ui.render(score_text, True, self.COLOR_TEXT)
        self.screen.blit(score_surf, (15, self.HEIGHT - score_surf.get_height() - 10))

        # Game Over / You Win text
        if self.game_over and not self.game_won:
            self._render_centered_text("GAME OVER", self.COLOR_HEART)
        elif self.game_won:
            self._render_centered_text("YOU WIN!", self.COLOR_GOAL)

    def _render_centered_text(self, text, color):
        text_surf = self.font_stage.render(text, True, color)
        text_rect = text_surf.get_rect(center=(self.WIDTH//2, self.HEIGHT//2))
        pygame.draw.rect(self.screen, (0,0,0,150), text_rect.inflate(20,20))
        self.screen.blit(text_surf, text_rect)

    def _draw_heart(self, x, y, size=12):
        # A simple heart shape using two circles and a triangle
        pygame.draw.circle(self.screen, self.COLOR_HEART, (x - size//2, y - size//4), size//2)
        pygame.draw.circle(self.screen, self.COLOR_HEART, (x + size//2, y - size//4), size//2)
        # gfxdraw requires integer coordinates
        pygame.gfxdraw.filled_trigon(
            self.screen, 
            x - size, y - size//4, 
            x + size, y - size//4, 
            x, int(y + size*0.75), 
            self.COLOR_HEART
        )

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "lives": self.lives,
            "stage": self.stage,
        }

    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # This block allows you to play the game directly
    os.environ["SDL_VIDEODRIVER"] = "x11" # Use 'x11' or 'windows' or 'mac'
    
    # Re-initialize pygame with video
    pygame.quit()
    pygame.init()

    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Arcade Hopper")
    clock = pygame.time.Clock()
    
    terminated = False
    truncated = False
    
    print(env.user_guide)
    
    while not terminated and not truncated:
        # --- Human Controls ---
        keys = pygame.key.get_pressed()
        movement = 0 # none
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]
        
        # --- Gym Step ---
        obs, reward, terminated, truncated, info = env.step(action)
        
        # --- Pygame Rendering ---
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        # --- Event Handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                print("Resetting environment.")
                obs, info = env.reset()

        clock.tick(env.FPS)
        
    print(f"Game Over! Final Info: {info}")
    env.close()