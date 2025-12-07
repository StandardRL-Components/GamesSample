import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T14:39:05.133155
# Source Brief: brief_00535.md
# Brief Index: 535
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    A Gymnasium environment where the player controls a cube, cloning momentum-conserving
    projectiles to destroy waves of geometric enemies. The game features a clean,
    neon-infused aesthetic with a focus on satisfying game feel and visual feedback.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Control a cube and clone momentum-conserving projectiles to destroy waves of geometric enemies in this neon-infused arcade shooter."
    )
    user_guide = (
        "Controls: Use arrow keys (↑↓←→) to move, press Space to clone a projectile, and use Shift to switch between fast and slow modes."
    )
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    FPS = 30
    MAX_STEPS = 1500 # Increased for more gameplay potential
    MAX_WAVES = 5

    # Colors
    COLOR_BG = (15, 18, 32)
    COLOR_GRID = (30, 35, 60)
    COLOR_PLAYER_SLOW = (0, 150, 255)
    COLOR_PLAYER_FAST = (255, 50, 100)
    COLOR_ENEMY = (50, 255, 120)
    COLOR_TEXT = (240, 240, 240)
    COLOR_PARTICLE = (220, 220, 220)
    
    # Player Physics
    PLAYER_SIZE = 20
    PLAYER_ACCEL = 0.8
    PLAYER_FRICTION = 0.92
    PLAYER_MAX_SPEED_SLOW = 4.0
    PLAYER_MAX_SPEED_FAST = 8.0

    # Projectile Physics
    PROJECTILE_SIZE = 12
    PROJECTILE_SPEED_BOOST_SLOW = 6.0
    PROJECTILE_SPEED_BOOST_FAST = 10.0
    PROJECTILE_COOLDOWN = 6 # frames

    # Enemy Physics
    ENEMY_SIZE = 24
    ENEMY_BASE_SPEED = 1.0
    ENEMY_SPEED_WAVE_INCREMENT = 0.25


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
        self.font_ui = pygame.font.SysFont("Consolas", 20, bold=True)
        self.font_state = pygame.font.SysFont("Consolas", 16)

        # Initialize state variables that are not reset every episode
        self.prev_shift_held = False
        
        # This will be properly initialized in reset()
        self.player_pos = pygame.math.Vector2(0, 0)
        self.player_vel = pygame.math.Vector2(0, 0)
        self.aim_direction = pygame.math.Vector2(0, -1)
        self.is_fast_mode = False
        self.projectiles = []
        self.enemies = []
        self.particles = []
        self.clone_cooldown_timer = 0
        self.steps = 0
        self.score = 0
        self.wave_number = 0
        self.enemy_current_speed = 0.0
        self.game_over = False

        self.reset()
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Player state
        self.player_pos = pygame.math.Vector2(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2)
        self.player_vel = pygame.math.Vector2(0, 0)
        self.aim_direction = pygame.math.Vector2(0, -1) # Default aim up
        self.is_fast_mode = False
        self.prev_shift_held = False

        # Game state
        self.steps = 0
        self.score = 0
        self.wave_number = 0
        self.enemy_current_speed = self.ENEMY_BASE_SPEED
        self.game_over = False
        
        # Entity lists
        self.projectiles.clear()
        self.enemies.clear()
        self.particles.clear()
        
        # Cooldowns
        self.clone_cooldown_timer = 0

        self._spawn_wave()
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            # If the game is over, do nothing and just return the final state
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        reward = 0.0

        # --- Handle Input and State Changes ---
        self._handle_input(action)

        # --- Update Game Logic ---
        self._update_player()
        self._update_projectiles()
        self._update_enemies()
        self._update_particles()
        
        # --- Handle Collisions and Rewards ---
        reward += self._handle_collisions()

        # --- Check for Wave Clear ---
        if not self.enemies:
            reward += 1.0 # Reward for clearing a wave
            self.score += 100 * self.wave_number
            if self.wave_number >= self.MAX_WAVES:
                self.game_over = True
                reward += 100.0 # Big reward for winning
            else:
                # sfx: wave_cleared_sound()
                self._spawn_wave()

        # --- Check Termination Conditions ---
        terminated = self.game_over or self.steps >= self.MAX_STEPS
        truncated = self.steps >= self.MAX_STEPS

        if self.game_over and not truncated: # Player died
            reward = -100.0
        
        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _handle_input(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        # Movement and Aiming
        move_vector = pygame.math.Vector2(0, 0)
        if movement == 1: # Up
            move_vector.y = -1
        elif movement == 2: # Down
            move_vector.y = 1
        elif movement == 3: # Left
            move_vector.x = -1
        elif movement == 4: # Right
            move_vector.x = 1
        
        if move_vector.length_squared() > 0:
            self.player_vel += move_vector.normalize() * self.PLAYER_ACCEL
            self.aim_direction = move_vector.normalize()

        # State Switching (Shift)
        if shift_held and not self.prev_shift_held:
            self.is_fast_mode = not self.is_fast_mode
            # sfx: state_switch_sound()
            # Add a visual effect for state switching
            for _ in range(15):
                self.particles.append(self._create_particle(self.player_pos, random.uniform(2, 5)))
        self.prev_shift_held = shift_held

        # Cloning (Space)
        if space_held and self.clone_cooldown_timer <= 0:
            self._clone_projectile()
            self.clone_cooldown_timer = self.PROJECTILE_COOLDOWN
            # sfx: clone_sound()

    def _update_player(self):
        # Apply friction and cap speed
        max_speed = self.PLAYER_MAX_SPEED_FAST if self.is_fast_mode else self.PLAYER_MAX_SPEED_SLOW
        self.player_vel *= self.PLAYER_FRICTION
        if self.player_vel.length() > max_speed:
            self.player_vel.scale_to_length(max_speed)
        
        self.player_pos += self.player_vel
        
        # Boundary checks
        player_rect = pygame.Rect(self.player_pos.x - self.PLAYER_SIZE / 2, self.player_pos.y - self.PLAYER_SIZE / 2, self.PLAYER_SIZE, self.PLAYER_SIZE)
        player_rect.clamp_ip(self.screen.get_rect())
        self.player_pos.x, self.player_pos.y = player_rect.centerx, player_rect.centery
        
        # Update cooldown
        if self.clone_cooldown_timer > 0:
            self.clone_cooldown_timer -= 1

    def _clone_projectile(self):
        boost_speed = self.PROJECTILE_SPEED_BOOST_FAST if self.is_fast_mode else self.PROJECTILE_SPEED_BOOST_SLOW
        
        # Projectile inherits player velocity and gets a boost in the aim direction
        proj_vel = self.player_vel + self.aim_direction * boost_speed
        
        # Color depends on player state
        color = self.COLOR_PLAYER_FAST if self.is_fast_mode else self.COLOR_PLAYER_SLOW

        self.projectiles.append({
            "pos": self.player_pos.copy(),
            "vel": proj_vel,
            "color": color,
            "rect": pygame.Rect(0, 0, self.PROJECTILE_SIZE, self.PROJECTILE_SIZE)
        })

    def _update_projectiles(self):
        screen_rect = self.screen.get_rect()
        for proj in self.projectiles[:]:
            proj["pos"] += proj["vel"]
            proj["rect"].center = proj["pos"]
            if not screen_rect.colliderect(proj["rect"]):
                self.projectiles.remove(proj)

    def _update_enemies(self):
        screen_rect = self.screen.get_rect().inflate(self.ENEMY_SIZE * 2, self.ENEMY_SIZE * 2)
        for enemy in self.enemies[:]:
            enemy["pos"] += enemy["vel"]
            enemy["rect"].center = enemy["pos"]
            if not screen_rect.colliderect(enemy["rect"]):
                self.enemies.remove(enemy)

    def _update_particles(self):
        for p in self.particles[:]:
            p["pos"] += p["vel"]
            p["life"] -= 1
            if p["life"] <= 0:
                self.particles.remove(p)

    def _handle_collisions(self):
        reward = 0.0

        # Projectile-Enemy collisions
        for proj in self.projectiles[:]:
            for enemy in self.enemies[:]:
                if proj["rect"].colliderect(enemy["rect"]):
                    # sfx: enemy_hit_sound()
                    self.score += 10
                    reward += 0.1
                    
                    # Create explosion particles
                    for _ in range(20):
                        self.particles.append(self._create_particle(enemy["pos"], random.uniform(1, 4)))
                    
                    if proj in self.projectiles: self.projectiles.remove(proj)
                    if enemy in self.enemies: self.enemies.remove(enemy)
                    break 

        # Player-Enemy collisions
        player_rect = pygame.Rect(0, 0, self.PLAYER_SIZE, self.PLAYER_SIZE)
        player_rect.center = self.player_pos
        for enemy in self.enemies:
            if player_rect.colliderect(enemy["rect"]):
                # sfx: player_death_sound()
                self.game_over = True
                # Create a larger explosion for player death
                for _ in range(50):
                    self.particles.append(self._create_particle(self.player_pos, random.uniform(2, 6)))
                break
        
        return reward

    def _spawn_wave(self):
        self.wave_number += 1
        self.enemy_current_speed = self.ENEMY_BASE_SPEED + (self.wave_number - 1) * self.ENEMY_SPEED_WAVE_INCREMENT
        num_enemies = 3 + self.wave_number * 2

        for _ in range(num_enemies):
            # Spawn from random edges
            edge = random.choice(["top", "bottom", "left", "right"])
            if edge == "top":
                pos = pygame.math.Vector2(random.uniform(0, self.SCREEN_WIDTH), -self.ENEMY_SIZE)
            elif edge == "bottom":
                pos = pygame.math.Vector2(random.uniform(0, self.SCREEN_WIDTH), self.SCREEN_HEIGHT + self.ENEMY_SIZE)
            elif edge == "left":
                pos = pygame.math.Vector2(-self.ENEMY_SIZE, random.uniform(0, self.SCREEN_HEIGHT))
            else: # right
                pos = pygame.math.Vector2(self.SCREEN_WIDTH + self.ENEMY_SIZE, random.uniform(0, self.SCREEN_HEIGHT))
            
            # Aim towards the center of the screen with some variance
            target_point = pygame.math.Vector2(
                self.SCREEN_WIDTH / 2 + random.uniform(-100, 100),
                self.SCREEN_HEIGHT / 2 + random.uniform(-100, 100)
            )
            vel = (target_point - pos).normalize() * self.enemy_current_speed

            self.enemies.append({
                "pos": pos,
                "vel": vel,
                "rect": pygame.Rect(0, 0, self.ENEMY_SIZE, self.ENEMY_SIZE)
            })

    def _create_particle(self, pos, speed):
        angle = random.uniform(0, 2 * math.pi)
        vel = pygame.math.Vector2(math.cos(angle), math.sin(angle)) * speed
        return {
            "pos": pos.copy(),
            "vel": vel,
            "life": random.randint(15, 30),
            "size": random.uniform(1, 4)
        }

    def _get_observation(self):
        # Clear screen with background
        self.screen.fill(self.COLOR_BG)
        
        # Render all game elements
        self._render_background()
        self._render_particles()
        self._render_enemies()
        self._render_projectiles()
        if not self.game_over:
            self._render_player()
        
        # Render UI overlay
        self._render_ui()
        
        # Convert to numpy array
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "wave": self.wave_number,
            "player_vel": (self.player_vel.x, self.player_vel.y),
            "is_fast_mode": self.is_fast_mode
        }

    # --- Rendering Methods ---

    def _render_glow(self, surface, color, rect, blur_radius):
        glow_color = pygame.Color(*color)
        glow_color.a = 20 # Low alpha for soft glow
        
        for i in range(blur_radius, 0, -2):
            glow_surf = pygame.Surface((rect.width + i*2, rect.height + i*2), pygame.SRCALPHA)
            pygame.draw.rect(glow_surf, glow_color, glow_surf.get_rect(), border_radius=int(i/2))
            surface.blit(glow_surf, (rect.x - i, rect.y - i), special_flags=pygame.BLEND_RGBA_ADD)

    def _render_background(self):
        for x in range(0, self.SCREEN_WIDTH, 40):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.SCREEN_HEIGHT), 1)
        for y in range(0, self.SCREEN_HEIGHT, 40):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.SCREEN_WIDTH, y), 1)

    def _render_player(self):
        player_color = self.COLOR_PLAYER_FAST if self.is_fast_mode else self.COLOR_PLAYER_SLOW
        player_rect = pygame.Rect(0, 0, self.PLAYER_SIZE, self.PLAYER_SIZE)
        player_rect.center = self.player_pos
        
        # Glow effect
        self._render_glow(self.screen, player_color, player_rect, 10)
        
        # Core shape
        pygame.draw.rect(self.screen, player_color, player_rect, border_radius=3)
        
        # Aim indicator
        aim_end = self.player_pos + self.aim_direction * (self.PLAYER_SIZE)
        pygame.draw.line(self.screen, (255, 255, 255), self.player_pos, aim_end, 2)

    def _render_projectiles(self):
        for proj in self.projectiles:
            rect = proj["rect"]
            self._render_glow(self.screen, proj["color"], rect, 8)
            pygame.draw.rect(self.screen, proj["color"], rect, border_radius=2)

    def _render_enemies(self):
        for enemy in self.enemies:
            rect = enemy["rect"]
            # Render as a triangle instead of a square for variety
            p1 = (rect.centerx, rect.top)
            p2 = (rect.left, rect.bottom)
            p3 = (rect.right, rect.bottom)
            
            # Glow effect for triangle
            glow_color = pygame.Color(*self.COLOR_ENEMY)
            glow_color.a = 20
            for i in range(8, 0, -2):
                glow_surf = pygame.Surface(rect.size, pygame.SRCALPHA)
                pygame.gfxdraw.aapolygon(glow_surf, [(p1[0]-rect.x, p1[1]-rect.y), (p2[0]-rect.x, p2[1]-rect.y), (p3[0]-rect.x, p3[1]-rect.y)], glow_color)
                glow_surf = pygame.transform.smoothscale(glow_surf, (rect.width + i*2, rect.height + i*2))
                self.screen.blit(glow_surf, (rect.x - i, rect.y - i), special_flags=pygame.BLEND_RGBA_ADD)

            # Core shape
            pygame.gfxdraw.aapolygon(self.screen, [p1, p2, p3], self.COLOR_ENEMY)
            pygame.gfxdraw.filled_polygon(self.screen, [p1, p2, p3], self.COLOR_ENEMY)

    def _render_particles(self):
        for p in self.particles:
            alpha = int(255 * (p["life"] / 30.0))
            color = (*self.COLOR_PARTICLE, alpha)
            if alpha > 0:
                # Using a surface for alpha blending
                particle_surf = pygame.Surface((p["size"]*2, p["size"]*2), pygame.SRCALPHA)
                pygame.draw.rect(particle_surf, color, particle_surf.get_rect())
                self.screen.blit(particle_surf, (int(p["pos"].x - p["size"]), int(p["pos"].y - p["size"])))

    def _render_ui(self):
        # Score
        score_text = self.font_ui.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))

        # Wave
        wave_text = self.font_ui.render(f"WAVE: {self.wave_number}/{self.MAX_WAVES}", True, self.COLOR_TEXT)
        self.screen.blit(wave_text, (self.SCREEN_WIDTH - wave_text.get_width() - 10, 10))
        
        # State
        state_str = "FAST" if self.is_fast_mode else "SLOW"
        state_color = self.COLOR_PLAYER_FAST if self.is_fast_mode else self.COLOR_PLAYER_SLOW
        state_text = self.font_state.render(f"MODE: {state_str}", True, state_color)
        self.screen.blit(state_text, (10, self.SCREEN_HEIGHT - state_text.get_height() - 10))

        # Cooldown indicator
        if self.clone_cooldown_timer > 0:
            cooldown_ratio = self.clone_cooldown_timer / self.PROJECTILE_COOLDOWN
            bar_width = 50
            bar_height = 5
            fill_width = int(bar_width * cooldown_ratio)
            
            bar_rect = pygame.Rect(self.SCREEN_WIDTH/2 - bar_width/2, self.SCREEN_HEIGHT - 20, bar_width, bar_height)
            fill_rect = pygame.Rect(bar_rect.x, bar_rect.y, fill_width, bar_height)
            
            pygame.draw.rect(self.screen, self.COLOR_GRID, bar_rect)
            pygame.draw.rect(self.screen, self.COLOR_TEXT, fill_rect)

    def close(self):
        pygame.quit()


if __name__ == '__main__':
    # This block allows you to play the game manually for testing
    env = GameEnv()
    obs, info = env.reset()
    
    # Override the screen to be a display surface
    env.screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Momentum Cloner")
    
    terminated = False
    truncated = False
    total_reward = 0
    
    print(GameEnv.user_guide)

    while not (terminated or truncated):
        # --- Human Input ---
        keys = pygame.key.get_pressed()
        movement = 0 # no-op
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]

        # --- Gym Step ---
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # --- Rendering ---
        # The observation is already the rendered frame, so we just need to display it
        # A more optimized way would be to have a separate render() method.
        rendered_frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(rendered_frame)
        env.screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        # --- Event Handling & Clock ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True
        
        env.clock.tick(env.FPS)

    print(f"Game Over. Final Score: {info['score']}. Total Reward: {total_reward:.2f}")
    env.close()