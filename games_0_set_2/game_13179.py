import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T19:31:29.773788
# Source Brief: brief_03179.md
# Brief Index: 3179
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

# Helper class for particles
class Particle:
    def __init__(self, pos, vel, color, life):
        self.pos = pygame.math.Vector2(pos)
        self.vel = pygame.math.Vector2(vel)
        self.color = color
        self.life = life
        self.max_life = life

    def update(self):
        self.pos += self.vel
        self.life -= 1
        self.vel *= 0.98  # Damping

    def draw(self, surface):
        if self.life > 0:
            alpha = int(255 * (self.life / self.max_life))
            radius = int(3 * (self.life / self.max_life))
            if radius > 0:
                # Simple glow effect
                glow_radius = int(radius * 1.5)
                glow_color = (*self.color, int(alpha * 0.5))
                pygame.gfxdraw.filled_circle(surface, int(self.pos.x), int(self.pos.y), glow_radius, glow_color)
                pygame.gfxdraw.filled_circle(surface, int(self.pos.x), int(self.pos.y), radius, (*self.color, alpha))


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Bounce a glowing orb through a neon world of platforms, collecting power-ups "
        "to maximize your score before time runs out."
    )
    user_guide = "Controls: Use ←→ arrow keys to move left and right. Press space to jump."
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Screen dimensions
        self.WIDTH, self.HEIGHT = 640, 400
        
        # EXACT spaces:
        self.observation_space = Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 36)
        self.small_font = pygame.font.Font(None, 24)
        
        # Colors
        self.COLOR_BG_TOP = (10, 0, 20)
        self.COLOR_BG_BOTTOM = (30, 0, 50)
        self.COLOR_PLAYER = (0, 150, 255)
        self.COLOR_PLATFORM = (40, 40, 60)
        self.COLOR_TEXT = (220, 220, 255)
        self.POWERUP_COLORS = {
            "double_jump": (0, 255, 100),
            "speed_boost": (255, 50, 50),
            "size_change": (255, 255, 0),
        }

        # Game constants
        self.GRAVITY = 0.4
        self.MAX_VEL_Y = 8
        self.PLAYER_ACCEL = 0.8
        self.PLAYER_FRICTION = 0.90
        self.JUMP_STRENGTH = 9.0
        self.BOUNCE_FACTOR = 0.6
        self.MAX_STEPS = 3000 # 100 seconds at 30 FPS logic update
        self.GAME_DURATION_SECONDS = 100
        
        # Initialize state variables
        self._init_state()

    def _init_state(self):
        """Initializes all game state variables. Called by __init__ and reset."""
        self.player_pos = pygame.math.Vector2(self.WIDTH / 2, self.HEIGHT - 50)
        self.player_vel = pygame.math.Vector2(0, 0)
        self.player_radius = 12
        self.is_grounded = False
        self.jumps_left = 1

        self.platforms = []
        self.powerups = []
        self.particles = []

        self.score = 0
        self.multiplier = 1
        self.steps = 0
        self.level_complexity = 0

        self.active_powerups = {
            "double_jump": 0,
            "speed_boost": 0,
            "size_change": 0,
        }
        self.powerup_spawn_timer = 0

        self.prev_space_held = False
        self.game_over = False

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self._init_state()
        self._generate_level()
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        self._handle_input(movement, space_held)
        self._update_physics()
        self._update_game_state()

        reward = self._calculate_reward()
        terminated = self.steps >= self.MAX_STEPS
        truncated = False
        if terminated:
            self.game_over = True
            reward += 10 * self.multiplier # Final bonus

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _handle_input(self, movement, space_held):
        # Horizontal movement
        current_accel = self.PLAYER_ACCEL * (1.5 if self.active_powerups["speed_boost"] > 0 else 1)
        if movement == 3:  # Left
            self.player_vel.x -= current_accel
        elif movement == 4:  # Right
            self.player_vel.x += current_accel

        # Jumping
        jump_pressed = space_held and not self.prev_space_held
        if jump_pressed and self.jumps_left > 0:
            self.player_vel.y = -self.JUMP_STRENGTH
            self.jumps_left -= 1
            self.is_grounded = False
            # sfx: jump.wav
            for _ in range(15):
                angle = self.np_random.uniform(math.pi * 1.25, math.pi * 1.75)
                speed = self.np_random.uniform(1, 3)
                vel = pygame.math.Vector2(math.cos(angle), math.sin(angle)) * speed
                self.particles.append(Particle(self.player_pos, vel, self.COLOR_PLAYER, 30))

        self.prev_space_held = space_held
    
    def _update_physics(self):
        # Apply gravity
        self.player_vel.y += self.GRAVITY
        self.player_vel.y = min(self.player_vel.y, self.MAX_VEL_Y)

        # Apply friction
        self.player_vel.x *= self.PLAYER_FRICTION
        if abs(self.player_vel.x) < 0.1:
            self.player_vel.x = 0

        # Update position
        self.player_pos += self.player_vel

        # Collision detection and resolution
        self.is_grounded = False
        player_rect = pygame.Rect(self.player_pos.x - self.player_radius, self.player_pos.y - self.player_radius, self.player_radius * 2, self.player_radius * 2)

        for plat in self.platforms:
            if player_rect.colliderect(plat):
                # Calculate overlap
                delta_x = self.player_pos.x - plat.centerx
                delta_y = self.player_pos.y - plat.centery
                
                # Push out
                overlap_x = (self.player_radius + plat.width / 2) - abs(delta_x)
                overlap_y = (self.player_radius + plat.height / 2) - abs(delta_y)

                if overlap_x > 0 and overlap_y > 0:
                    if overlap_y < overlap_x:
                        # Vertical collision
                        if delta_y > 0: # From top
                            self.player_pos.y += overlap_y
                            if self.player_vel.y < 0: self.player_vel.y = 0
                        else: # From bottom
                            self.player_pos.y -= overlap_y
                            if self.player_vel.y > 0:
                                self.player_vel.y *= -self.BOUNCE_FACTOR
                                if abs(self.player_vel.y) < 1: self.player_vel.y = 0
                                self.is_grounded = True
                                self.jumps_left = 1 + (1 if self.active_powerups["double_jump"] > 0 else 0)
                    else:
                        # Horizontal collision
                        if delta_x > 0: # From left
                            self.player_pos.x += overlap_x
                        else: # From right
                            self.player_pos.x -= overlap_x
                        self.player_vel.x *= -self.BOUNCE_FACTOR
        
        # Boundary checks
        if self.player_pos.x - self.player_radius < 0:
            self.player_pos.x = self.player_radius
            self.player_vel.x *= -self.BOUNCE_FACTOR
        if self.player_pos.x + self.player_radius > self.WIDTH:
            self.player_pos.x = self.WIDTH - self.player_radius
            self.player_vel.x *= -self.BOUNCE_FACTOR
        if self.player_pos.y - self.player_radius < 0:
            self.player_pos.y = self.player_radius
            self.player_vel.y *= -self.BOUNCE_FACTOR
        if self.player_pos.y + self.player_radius > self.HEIGHT:
            self.player_pos.y = self.HEIGHT - self.player_radius
            self.player_vel.y *= -self.BOUNCE_FACTOR
            self.is_grounded = True
            self.jumps_left = 1 + (1 if self.active_powerups["double_jump"] > 0 else 0)

    def _update_game_state(self):
        self.steps += 1

        # Update complexity every 20 seconds (600 steps at 30fps)
        if self.steps > 0 and self.steps % 600 == 0:
            self.level_complexity = min(5, self.level_complexity + 1)
            self._generate_level()
            # sfx: level_up.wav

        # Update powerup timers
        for key in self.active_powerups:
            if self.active_powerups[key] > 0:
                self.active_powerups[key] -= 1
        
        # Update player size based on powerup
        target_radius = 12
        if self.active_powerups["size_change"] > 0:
            # Size change alternates between big and small
            if (self.active_powerups["size_change"] // 150) % 2 == 1:
                target_radius = 18 # Big
            else:
                target_radius = 8 # Small
        self.player_radius += (target_radius - self.player_radius) * 0.1 # Smooth transition

        # Update particles
        self.particles = [p for p in self.particles if p.life > 0]
        for p in self.particles:
            p.update()

        # Update powerups
        for p_up in self.powerups:
            p_up['angle'] = (p_up['angle'] + 2) % 360
            p_up['pulse'] = math.sin(self.steps * 0.1 + p_up['angle']) * 2

        # Spawn new powerups
        self.powerup_spawn_timer -= 1
        if len(self.powerups) < 3 and self.powerup_spawn_timer <= 0:
            self._spawn_powerup()
            self.powerup_spawn_timer = 90 # 3 seconds

    def _calculate_reward(self):
        reward = 0.01  # Small survival reward

        # Check for powerup collection
        collected_indices = []
        player_rect = pygame.Rect(self.player_pos.x - self.player_radius, self.player_pos.y - self.player_radius, self.player_radius * 2, self.player_radius * 2)
        
        for i, p_up in enumerate(self.powerups):
            p_up_rect = pygame.Rect(p_up['pos'][0] - 10, p_up['pos'][1] - 10, 20, 20)
            if player_rect.colliderect(p_up_rect):
                collected_indices.append(i)
                
                # Apply effect
                p_type = p_up['type']
                self.active_powerups[p_type] = 150 # 5 seconds at 30fps
                if p_type == "double_jump":
                    self.jumps_left = 2
                
                # Scoring
                self.score += 10 * self.multiplier
                self.multiplier += 1
                reward += 5.0

                # Visual/Audio Feedback
                # sfx: powerup_collect.wav
                for _ in range(30):
                    angle = self.np_random.uniform(0, 2 * math.pi)
                    speed = self.np_random.uniform(2, 5)
                    vel = pygame.math.Vector2(math.cos(angle), math.sin(angle)) * speed
                    self.particles.append(Particle(p_up['pos'], vel, p_up['color'], 40))

        # Remove collected powerups
        if collected_indices:
            self.powerups = [p for i, p in enumerate(self.powerups) if i not in collected_indices]
        
        return reward

    def _generate_level(self):
        self.platforms.clear()
        
        num_platforms = 5 + self.level_complexity * 2
        min_w, max_w = max(80, 150 - self.level_complexity * 15), max(120, 300 - self.level_complexity * 30)
        min_h, max_h = 10, 20

        for _ in range(num_platforms):
            while True:
                w = self.np_random.uniform(min_w, max_w)
                h = self.np_random.uniform(min_h, max_h)
                x = self.np_random.uniform(0, self.WIDTH - w)
                y = self.np_random.uniform(50, self.HEIGHT - 50)
                
                new_plat = pygame.Rect(x, y, w, h)
                
                # Ensure spawn area is clear
                spawn_area = pygame.Rect(self.WIDTH/2 - 50, self.HEIGHT - 100, 100, 100)
                if not new_plat.colliderect(spawn_area):
                    self.platforms.append(new_plat)
                    break
    
    def _spawn_powerup(self):
        powerup_type = self.np_random.choice(list(self.POWERUP_COLORS.keys()))
        
        for _ in range(50): # Try 50 times to find a good spot
            pos = (self.np_random.uniform(50, self.WIDTH - 50), self.np_random.uniform(50, self.HEIGHT - 50))
            
            # Check for clearance from platforms
            is_safe = True
            test_rect = pygame.Rect(pos[0] - 20, pos[1] - 20, 40, 40)
            for plat in self.platforms:
                if plat.colliderect(test_rect):
                    is_safe = False
                    break
            
            if is_safe:
                self.powerups.append({
                    'pos': pos,
                    'type': powerup_type,
                    'color': self.POWERUP_COLORS[powerup_type],
                    'angle': self.np_random.uniform(0, 360),
                    'pulse': 0
                })
                break
    
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
                int(self.COLOR_BG_TOP[2] * (1 - interp) + self.COLOR_BG_BOTTOM[2] * interp)
            )
            pygame.draw.line(self.screen, color, (0, y), (self.WIDTH, y))
    
    def _render_game(self):
        # Render platforms
        for plat in self.platforms:
            pygame.draw.rect(self.screen, self.COLOR_PLATFORM, plat, border_radius=3)

        # Render particles
        for p in self.particles:
            p.draw(self.screen)

        # Render powerups
        for p_up in self.powerups:
            pos = p_up['pos']
            color = p_up['color']
            radius = 10 + p_up['pulse']
            
            # Rotating square
            points = []
            for i in range(4):
                angle = math.radians(p_up['angle'] + 90 * i)
                x = pos[0] + math.cos(angle) * radius * 1.414
                y = pos[1] + math.sin(angle) * radius * 1.414
                points.append((x, y))
            
            # Draw with glow
            pygame.gfxdraw.filled_polygon(self.screen, points, (*color, 50))
            pygame.gfxdraw.aapolygon(self.screen, points, color)

        # Render player
        pos = (int(self.player_pos.x), int(self.player_pos.y))
        radius = int(self.player_radius)
        
        # Glow effect
        for i in range(radius // 2, 0, -1):
            alpha = int(100 * (1 - (i / (radius // 2))**2))
            glow_color = (*self.COLOR_PLAYER, alpha)
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], radius + i, glow_color)
        
        pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], radius, self.COLOR_PLAYER)
        pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], radius, self.COLOR_PLAYER)

    def _render_ui(self):
        # Score and Multiplier
        score_text = self.font.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        mult_text = self.small_font.render(f"x{self.multiplier}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))
        self.screen.blit(mult_text, (score_text.get_width() + 15, 18))

        # Timer
        time_left = max(0, self.GAME_DURATION_SECONDS - (self.steps / (self.MAX_STEPS / self.GAME_DURATION_SECONDS)))
        timer_text = self.font.render(f"TIME: {time_left:.1f}", True, self.COLOR_TEXT)
        self.screen.blit(timer_text, (self.WIDTH - timer_text.get_width() - 10, 10))

        # Powerup status
        y_offset = 45
        for p_type, duration in self.active_powerups.items():
            if duration > 0:
                text = self.small_font.render(p_type.replace('_', ' ').upper(), True, self.POWERUP_COLORS[p_type])
                self.screen.blit(text, (10, y_offset))
                
                bar_width = 100
                filled_width = int(bar_width * (duration / 150.0))
                pygame.draw.rect(self.screen, self.POWERUP_COLORS[p_type], (text.get_width() + 15, y_offset + 4, filled_width, 10))
                pygame.draw.rect(self.screen, self.COLOR_PLATFORM, (text.get_width() + 15, y_offset + 4, bar_width, 10), 1)

                y_offset += 25

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "multiplier": self.multiplier,
            "level_complexity": self.level_complexity,
        }

    def close(self):
        pygame.quit()
        
if __name__ == "__main__":
    # This block is for human play and is not part of the Gymnasium environment
    # It will not be run by the evaluator
    try:
        env = GameEnv()
        obs, info = env.reset()
        
        # Pygame setup for human play
        screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
        pygame.display.set_caption("Recursive Bouncer")
        clock = pygame.time.Clock()
        
        running = True
        total_reward = 0
        
        while running:
            movement = 0 # None
            space = 0
            shift = 0

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                    obs, info = env.reset()
                    total_reward = 0

            keys = pygame.key.get_pressed()
            if keys[pygame.K_LEFT] or keys[pygame.K_a]:
                movement = 3
            elif keys[pygame.K_RIGHT] or keys[pygame.K_d]:
                movement = 4
            
            if keys[pygame.K_SPACE]:
                space = 1
            
            if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
                shift = 1

            action = [movement, space, shift]
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward

            if terminated or truncated:
                print(f"Episode finished. Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
                obs, info = env.reset()
                total_reward = 0
                
            # Render to the display
            surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
            screen.blit(surf, (0, 0))
            pygame.display.flip()
            
            clock.tick(30) # Limit to 30 FPS for human play

    finally:
        env.close()