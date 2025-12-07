import gymnasium as gym
import os
import pygame
import numpy as np
import pygame.gfxdraw
import math
import random
from gymnasium.spaces import MultiDiscrete
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T15:46:56.881491
# Source Brief: brief_01191.md
# Brief Index: 1191
# """import gymnasium as gym
class GameEnv(gym.Env):
    """
    A Gymnasium environment where the player controls a laser beam to deflect incoming projectiles.
    The goal is to deflect 25 projectiles within 60 seconds. The game features a minimalist,
    neon visual style with smooth animations and particle effects.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Control a rotating laser beam to deflect incoming projectiles. "
        "Deflect 25 projectiles before time runs out to win."
    )
    user_guide = "Controls: Use ← and → arrow keys to rotate the laser beam."
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    FPS = 30
    CENTER = pygame.Vector2(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2)

    # Game parameters
    GAME_DURATION_SECONDS = 60
    MAX_STEPS = GAME_DURATION_SECONDS * FPS
    WIN_SCORE = 25
    LASER_LENGTH = 300
    LASER_BASE_ROTATION_DEG_PER_STEP = 2.0
    LASER_SPEED_INCREASE_FACTOR = 1.05

    # Colors
    COLOR_BG = (15, 15, 25)
    COLOR_LASER_MAIN = (100, 200, 255)
    COLOR_LASER_GLOW = (50, 100, 200, 100)
    COLOR_PROJECTILE_MAIN = (255, 80, 80)
    COLOR_PROJECTILE_GLOW = (200, 50, 50, 100)
    COLOR_EXPLOSION = (255, 255, 100)
    COLOR_UI_TEXT = (220, 220, 220)
    COLOR_EMITTER = (200, 220, 255)

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        self.render_mode = render_mode

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("Consolas", 24, bold=True)
        self.font_game_over = pygame.font.SysFont("Consolas", 50, bold=True)

        # --- State Variables ---
        # These are initialized in reset()
        self.steps = 0
        self.score = 0
        self.time_left = 0.0
        self.game_over = False
        self.win_status = False

        self.laser_angle = 0.0
        self.laser_rotation_speed_multiplier = 1.0
        
        self.projectiles = []
        self.particles = []
        
        self.projectile_spawn_timer = 0
        self.projectile_spawn_rate = 1.5  # seconds
        self.projectile_min_speed = 100.0
        self.projectile_max_speed = 150.0

        self._background_stars = self._create_starfield(100)
        
        # Initialize state
        # self.reset() is called by the environment wrapper

    def _create_starfield(self, num_stars):
        stars = []
        for _ in range(num_stars):
            x = random.randint(0, self.SCREEN_WIDTH)
            y = random.randint(0, self.SCREEN_HEIGHT)
            size = random.randint(1, 2)
            brightness = random.randint(50, 150)
            stars.append(((x, y), size, (brightness, brightness, brightness)))
        return stars

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.time_left = self.GAME_DURATION_SECONDS
        self.game_over = False
        self.win_status = False

        self.laser_angle = self.np_random.uniform(0, 360)
        self.laser_rotation_speed_multiplier = 1.0

        self.projectiles.clear()
        self.particles.clear()
        
        self.projectile_spawn_timer = 0
        self.projectile_spawn_rate = 1.5
        self.projectile_min_speed = 100.0
        self.projectile_max_speed = 150.0

        # Spawn initial projectiles
        for _ in range(2):
            self._spawn_projectile()

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            # If the game is already over, do nothing and return the final state
            reward = 0.0
            terminated = True
            truncated = True # Game is over, so it's also truncated
            return self._get_observation(), reward, terminated, truncated, self._get_info()

        self.steps += 1
        self.time_left -= 1.0 / self.FPS

        movement = action[0]

        # 1. Update Laser Rotation
        rotation_this_step = self.LASER_BASE_ROTATION_DEG_PER_STEP * self.laser_rotation_speed_multiplier
        if movement == 3:  # Left (CCW)
            self.laser_angle -= rotation_this_step
        elif movement == 4:  # Right (CW)
            self.laser_angle += rotation_this_step
        self.laser_angle %= 360

        # 2. Update Game Entities
        self._update_projectiles()
        self._update_particles()
        
        # 3. Handle Spawning and Difficulty
        self._handle_spawning()
        self._scale_difficulty()
        
        # 4. Collision Detection and Event Rewards
        deflected_this_step = self._handle_collisions()
        reward = 1.0 * deflected_this_step
        if deflected_this_step > 0:
            # Sound: deflection_sound.play()
            self.score += deflected_this_step
            self.laser_rotation_speed_multiplier = self.LASER_SPEED_INCREASE_FACTOR ** self.score

        # 5. Check for Win/Loss Conditions
        terminated = False
        truncated = False
        if self.score >= self.WIN_SCORE:
            # Sound: win_sound.play()
            reward = 100.0
            terminated = True
            self.game_over = True
            self.win_status = True
        elif self.time_left <= 0 or self.steps >= self.MAX_STEPS:
            # Sound: lose_sound.play()
            reward = -100.0
            terminated = True # Time limit is a terminal condition
            self.game_over = True
        
        # 6. Continuous Aiming Reward (only on non-terminal steps)
        if not terminated:
            reward += self._calculate_aiming_reward()

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _spawn_projectile(self):
        edge = self.np_random.integers(4)
        if edge == 0:  # Top
            pos = pygame.Vector2(self.np_random.uniform(0, self.SCREEN_WIDTH), -20)
        elif edge == 1:  # Bottom
            pos = pygame.Vector2(self.np_random.uniform(0, self.SCREEN_WIDTH), self.SCREEN_HEIGHT + 20)
        elif edge == 2:  # Left
            pos = pygame.Vector2(-20, self.np_random.uniform(0, self.SCREEN_HEIGHT))
        else:  # Right
            pos = pygame.Vector2(self.SCREEN_WIDTH + 20, self.np_random.uniform(0, self.SCREEN_HEIGHT))
            
        direction = (self.CENTER - pos).normalize()
        speed = self.np_random.uniform(self.projectile_min_speed, self.projectile_max_speed)
        
        osc_frequency = self.np_random.uniform(2, 5) # Radians per second
        osc_phase = self.np_random.uniform(0, 2 * math.pi)

        self.projectiles.append({
            'pos': pos, 'dir': direction, 'base_speed': speed,
            'osc_freq': osc_frequency, 'osc_phase': osc_phase
        })

    def _update_projectiles(self):
        dt = 1.0 / self.FPS
        projectiles_to_keep = []
        for p in self.projectiles:
            p['osc_phase'] += p['osc_freq'] * dt
            speed_multiplier = 1.0 + 0.5 * math.sin(p['osc_phase'])
            current_speed = p['base_speed'] * speed_multiplier
            p['pos'] += p['dir'] * current_speed * dt
            
            if p['pos'].distance_to(self.CENTER) < 10:
                # Sound: miss_sound.play()
                pass # Projectile missed and is removed
            else:
                projectiles_to_keep.append(p)
        self.projectiles = projectiles_to_keep

    def _handle_spawning(self):
        self.projectile_spawn_timer -= 1.0 / self.FPS
        if self.projectile_spawn_timer <= 0:
            self._spawn_projectile()
            self.projectile_spawn_rate = max(0.3, 1.5 - (self.steps / self.MAX_STEPS) * 1.2)
            self.projectile_spawn_timer = self.projectile_spawn_rate

    def _scale_difficulty(self):
        if self.steps > 0 and self.steps % (10 * self.FPS) == 0:
            self.projectile_max_speed += 10.0
            self.projectile_min_speed += 5.0

    def _handle_collisions(self):
        deflected_count = 0
        laser_rad = math.radians(self.laser_angle)
        laser_dir = pygame.Vector2(math.cos(laser_rad), math.sin(laser_rad))
        
        projectiles_to_keep = []
        for p in self.projectiles:
            vec_to_proj = p['pos'] - self.CENTER
            dist_to_center = vec_to_proj.length()
            
            if 10 < dist_to_center < self.LASER_LENGTH and vec_to_proj.length() > 0:
                proj_dir = vec_to_proj.normalize()
                dot_product = laser_dir.dot(proj_dir)
                if dot_product > 0.995: # Angle diff < ~5.7 degrees
                    self._spawn_explosion(p['pos'], self.COLOR_EXPLOSION)
                    deflected_count += 1
                    continue
            
            projectiles_to_keep.append(p)
            
        self.projectiles = projectiles_to_keep
        return deflected_count

    def _calculate_aiming_reward(self):
        if not self.projectiles: return 0.0

        laser_rad = math.radians(self.laser_angle)
        laser_dir = pygame.Vector2(math.cos(laser_rad), math.sin(laser_rad))

        closest_proj = min(self.projectiles, key=lambda p: p['pos'].distance_to(self.CENTER))
        
        vec_to_proj = closest_proj['pos'] - self.CENTER
        if vec_to_proj.length() > 0:
            proj_dir = vec_to_proj.normalize()
            dot_product = laser_dir.dot(proj_dir)
            if dot_product > 0.98: # Angle diff < ~11.5 degrees
                return 0.1
        return 0.0

    def _spawn_explosion(self, position, color):
        num_particles = 30
        for _ in range(num_particles):
            vel = pygame.Vector2(self.np_random.uniform(-1, 1), self.np_random.uniform(-1, 1))
            if vel.length() > 0: vel.normalize_ip()
            vel *= self.np_random.uniform(50, 200)
            lifespan = self.np_random.uniform(0.3, 0.8)
            self.particles.append({
                'pos': position.copy(), 'vel': vel, 'lifespan': lifespan,
                'max_lifespan': lifespan, 'color': color
            })

    def _update_particles(self):
        dt = 1.0 / self.FPS
        self.particles = [p for p in self.particles if p['lifespan'] > 0]
        for p in self.particles:
            p['pos'] += p['vel'] * dt
            p['vel'] *= 0.95
            p['lifespan'] -= dt

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        for star in self._background_stars:
            pygame.draw.circle(self.screen, star[2], star[0], star[1])

        for p in self.particles:
            life_ratio = max(0, p['lifespan'] / p['max_lifespan'])
            color = tuple(int(c * life_ratio) for c in p['color'])
            pygame.draw.circle(self.screen, color, (int(p['pos'].x), int(p['pos'].y)), int(life_ratio * 3))

        for p in self.projectiles:
            pos_int = (int(p['pos'].x), int(p['pos'].y))
            pygame.gfxdraw.filled_circle(self.screen, pos_int[0], pos_int[1], 10, self.COLOR_PROJECTILE_GLOW)
            pygame.gfxdraw.filled_circle(self.screen, pos_int[0], pos_int[1], 6, self.COLOR_PROJECTILE_MAIN)
            pygame.gfxdraw.aacircle(self.screen, pos_int[0], pos_int[1], 6, self.COLOR_PROJECTILE_MAIN)
            
        pygame.gfxdraw.filled_circle(self.screen, int(self.CENTER.x), int(self.CENTER.y), 15, self.COLOR_EMITTER)
        pygame.gfxdraw.aacircle(self.screen, int(self.CENTER.x), int(self.CENTER.y), 15, self.COLOR_EMITTER)

        laser_rad = math.radians(self.laser_angle)
        end_pos = self.CENTER + pygame.Vector2(math.cos(laser_rad), math.sin(laser_rad)) * self.LASER_LENGTH
        pygame.draw.line(self.screen, self.COLOR_LASER_GLOW, self.CENTER, end_pos, width=10)
        pygame.draw.aaline(self.screen, self.COLOR_LASER_MAIN, self.CENTER, end_pos, blend=1)

    def _render_ui(self):
        score_text = self.font_ui.render(f"DEFLECTED: {self.score}/{self.WIN_SCORE}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (10, 10))

        time_text = self.font_ui.render(f"TIME: {max(0, self.time_left):.1f}", True, self.COLOR_UI_TEXT)
        time_rect = time_text.get_rect(topright=(self.SCREEN_WIDTH - 10, 10))
        self.screen.blit(time_text, time_rect)

        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            self.screen.blit(overlay, (0, 0))
            
            msg, color = ("VICTORY", self.COLOR_LASER_MAIN) if self.win_status else ("TIME UP", self.COLOR_PROJECTILE_MAIN)
            game_over_text = self.font_game_over.render(msg, True, color)
            text_rect = game_over_text.get_rect(center=self.CENTER)
            self.screen.blit(game_over_text, text_rect)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "time_left": self.time_left, "win": self.win_status}
        
    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # This block is for human play and debugging, and will not be run by the evaluation server.
    # Un-comment the os.environ line to run with a visible display.
    # os.environ["SDL_VIDEODRIVER"] = "x11"
    
    env = GameEnv()
    obs, info = env.reset()
    
    # We need to create a display for human play
    pygame.display.init()
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Laser Defender")
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0.0
    
    while running:
        action = [0, 0, 0] # [movement, space, shift]
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                running = False
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT] or keys[pygame.K_a]: action[0] = 3
        elif keys[pygame.K_RIGHT] or keys[pygame.K_d]: action[0] = 4
            
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # The observation is the game's rendered screen. We make a surface from it to draw.
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
            pygame.time.wait(3000)
            obs, info = env.reset()
            total_reward = 0.0

        clock.tick(env.FPS)
        
    env.close()
    pygame.quit()