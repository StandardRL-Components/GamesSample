import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T18:52:21.672341
# Source Brief: brief_03059.md
# Brief Index: 3059
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Trap mischievous enemies in magical bubbles before they reach the bottom. "
        "Aim your slingshot, choose your bubble size, and clear each level to win."
    )
    user_guide = (
        "Controls: Use ←→ arrow keys to aim your slingshot. Press space to launch a bubble "
        "and shift to cycle bubble size."
    )
    auto_advance = True
    
    # --- Constants ---
    # Colors
    COLOR_BG = (20, 25, 40)
    COLOR_WALLS = (40, 50, 80)
    COLOR_DANGER = (80, 20, 30, 100)
    COLOR_PLAYER = (255, 255, 255)
    COLOR_ENEMY = (255, 80, 80)
    COLOR_TEXT = (220, 220, 240)
    BUBBLE_COLORS = [
        (50, 150, 255), (100, 255, 100), (255, 100, 150), (255, 200, 50)
    ]
    
    # Screen Dimensions
    WIDTH, HEIGHT = 640, 400
    
    # Game Parameters
    MAX_STEPS = 1000
    DANGER_ZONE_HEIGHT = 50
    INITIAL_BUBBLES = 20
    AIM_SPEED = math.pi / 60  # Radians per step
    MIN_AIM_ANGLE = math.pi * 1.1
    MAX_AIM_ANGLE = math.pi * 1.9
    BUBBLE_LIFETIME = 300 # steps
    BUBBLE_SIZES = [15, 25, 35] # small, medium, large radii
    BUBBLE_SPEED_MOD = [1.5, 1.0, 0.7] # speed modifier for each size
    
    class Bubble:
        def __init__(self, pos, vel, radius, color, lifetime):
            self.pos = pygame.Vector2(pos)
            self.vel = pygame.Vector2(vel)
            self.radius = radius
            self.color = color
            self.lifetime = lifetime
            self.trapped_enemies = []
            self.has_trapped = False

    class Enemy:
        def __init__(self, pos, vel, radius=10):
            self.pos = pygame.Vector2(pos)
            self.vel = pygame.Vector2(vel)
            self.radius = radius
            self.state = "patrolling" # "patrolling" or "trapped"
            self.trapped_in = None

    class Particle:
        def __init__(self, pos, vel, size, color, lifetime):
            self.pos = pygame.Vector2(pos)
            self.vel = pygame.Vector2(vel)
            self.size = size
            self.color = color
            self.lifetime = lifetime
            self.max_lifetime = lifetime

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont("monospace", 16)
        self.font_large = pygame.font.SysFont("monospace", 32, bold=True)
        
        self.slingshot_pos = pygame.Vector2(self.WIDTH // 2, self.HEIGHT - self.DANGER_ZONE_HEIGHT // 2)
        
        self.level = 1
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.reward_buffer = 0.0
        
        self.enemies = []
        self.bubbles = []
        self.particles = []

        self.aim_angle = math.pi * 1.5
        self.current_bubble_size_idx = 0
        self.bubbles_left = self.INITIAL_BUBBLES
        
        self.prev_space_held = False
        self.prev_shift_held = False
        
        # self.reset() is called by the environment wrapper
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.reward_buffer = 0.0
        
        self.enemies.clear()
        self.bubbles.clear()
        self.particles.clear()
        
        self.aim_angle = math.pi * 1.5
        self.current_bubble_size_idx = 0
        self.bubbles_left = self.INITIAL_BUBBLES
        
        self.prev_space_held = False
        self.prev_shift_held = False
        
        self._spawn_enemies()
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        self.reward_buffer = -0.01 # Cost of existing

        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        self._handle_input(movement, space_held, shift_held)
        
        self._update_bubbles()
        self._update_enemies()
        self._update_particles()
        self._handle_collisions()

        terminated = self._check_termination()
        reward = self.reward_buffer
        self.score += reward
        
        # The truncated value is handled by the TimeLimit wrapper
        truncated = self.steps >= self.MAX_STEPS
        
        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )
        
    def _spawn_enemies(self):
        num_enemies = 1 + self.level // 2
        speed = 1.0 + (self.level // 5) * 0.5
        
        for i in range(num_enemies):
            x = self.np_random.uniform(50, self.WIDTH - 50)
            y = self.np_random.uniform(50, self.HEIGHT - 150)
            vx = self.np_random.choice([-1, 1]) * speed
            vy = self.np_random.choice([-1, 1]) * speed if self.np_random.random() < 0.5 else 0
            if vy == 0 and vx == 0: vx = speed # Ensure movement
            
            self.enemies.append(self.Enemy(pos=(x, y), vel=(vx, vy)))
            
    def _handle_input(self, movement, space_held, shift_held):
        # Aiming (1 & 3 aim left, 2 & 4 aim right)
        if movement == 1 or movement == 3: self.aim_angle -= self.AIM_SPEED # Left
        if movement == 2 or movement == 4: self.aim_angle += self.AIM_SPEED # Right
        self.aim_angle = np.clip(self.aim_angle, self.MIN_AIM_ANGLE, self.MAX_AIM_ANGLE)

        # Cycle bubble size on button press (rising edge)
        if shift_held and not self.prev_shift_held:
            self.current_bubble_size_idx = (self.current_bubble_size_idx + 1) % len(self.BUBBLE_SIZES)

        # Launch bubble on button press (rising edge)
        if space_held and not self.prev_space_held and self.bubbles_left > 0:
            self._launch_bubble()

        self.prev_space_held = space_held
        self.prev_shift_held = shift_held
        
    def _launch_bubble(self):
        self.bubbles_left -= 1
        radius = self.BUBBLE_SIZES[self.current_bubble_size_idx]
        speed_mod = self.BUBBLE_SPEED_MOD[self.current_bubble_size_idx]
        launch_speed = 6 * speed_mod
        
        vel = pygame.Vector2(math.cos(self.aim_angle), math.sin(self.aim_angle)) * launch_speed
        pos = self.slingshot_pos + vel.normalize() * (radius + 10)
        color_idx = self.np_random.integers(0, len(self.BUBBLE_COLORS))
        color = self.BUBBLE_COLORS[color_idx]
        
        new_bubble = self.Bubble(pos, vel, radius, color, self.BUBBLE_LIFETIME)
        self.bubbles.append(new_bubble)
        
    def _update_bubbles(self):
        bubbles_to_remove = []
        for bubble in self.bubbles:
            bubble.pos += bubble.vel
            bubble.lifetime -= 1
            
            # Wall bounces
            if bubble.pos.x - bubble.radius < 0 or bubble.pos.x + bubble.radius > self.WIDTH:
                bubble.vel.x *= -1
                bubble.pos.x = np.clip(bubble.pos.x, bubble.radius, self.WIDTH - bubble.radius)
            if bubble.pos.y - bubble.radius < 0:
                bubble.vel.y *= -1
                bubble.pos.y = np.clip(bubble.pos.y, bubble.radius, self.HEIGHT - bubble.radius)
            
            # Pop bubble if lifetime ends
            if bubble.lifetime <= 0:
                bubbles_to_remove.append(bubble)
                if bubble.has_trapped:
                    for enemy in bubble.trapped_enemies:
                        self.reward_buffer += 1.0 # Defeated an enemy
                        self.enemies.remove(enemy)
                else:
                    self.reward_buffer -= 1.0 # Wasted bubble
                self._create_particles(bubble.pos, bubble.color, 20)
        
        self.bubbles = [b for b in self.bubbles if b not in bubbles_to_remove]
        
    def _update_enemies(self):
        for enemy in self.enemies:
            if enemy.state == "patrolling":
                enemy.pos += enemy.vel
                if enemy.pos.x - enemy.radius < 0 or enemy.pos.x + enemy.radius > self.WIDTH:
                    enemy.vel.x *= -1
                if enemy.pos.y - enemy.radius < 0 or enemy.pos.y + enemy.radius > self.HEIGHT - self.DANGER_ZONE_HEIGHT:
                    enemy.vel.y *= -1
            elif enemy.state == "trapped":
                enemy.pos = enemy.trapped_in.pos # Follow the bubble

    def _update_particles(self):
        particles_to_remove = []
        for p in self.particles:
            p.pos += p.vel
            p.lifetime -= 1
            p.size = max(0, p.size - 0.1)
            if p.lifetime <= 0:
                particles_to_remove.append(p)
        self.particles = [p for p in self.particles if p not in particles_to_remove]
        
    def _handle_collisions(self):
        # Bubble-Enemy collisions
        for bubble in self.bubbles:
            for enemy in self.enemies:
                if enemy.state == "patrolling":
                    dist = bubble.pos.distance_to(enemy.pos)
                    if dist < bubble.radius + enemy.radius:
                        enemy.state = "trapped"
                        enemy.trapped_in = bubble
                        bubble.trapped_enemies.append(enemy)
                        bubble.has_trapped = True
                        self.reward_buffer += 0.1 # Trap reward

        # Bubble-Bubble collisions
        for i in range(len(self.bubbles)):
            for j in range(i + 1, len(self.bubbles)):
                b1, b2 = self.bubbles[i], self.bubbles[j]
                dist_vec = b1.pos - b2.pos
                if dist_vec.length_squared() == 0: continue # Avoid division by zero
                dist = dist_vec.length()
                if dist < b1.radius + b2.radius:
                    overlap = (b1.radius + b2.radius) - dist
                    normal = dist_vec.normalize()
                    
                    b1.pos += normal * overlap / 2
                    b2.pos -= normal * overlap / 2
                    
                    v1_dot_normal = b1.vel.dot(normal)
                    v2_dot_normal = b2.vel.dot(normal)
                    
                    b1.vel += (v2_dot_normal - v1_dot_normal) * normal
                    b2.vel += (v1_dot_normal - v2_dot_normal) * normal

    def _create_particles(self, pos, color, count):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = pygame.Vector2(math.cos(angle), math.sin(angle)) * speed
            size = self.np_random.uniform(2, 6)
            lifetime = self.np_random.integers(15, 30)
            self.particles.append(self.Particle(pos, vel, size, color, lifetime))

    def _check_termination(self):
        if self.game_over: return True
        
        # Win condition
        if not self.enemies:
            self.reward_buffer += 10.0
            self.game_over = True
            self.level += 1 # Progress to next level
            return True
            
        # Loss conditions
        if self.bubbles_left <= 0 and not self.bubbles:
            self.reward_buffer -= 10.0
            self.game_over = True
            return True
        
        for enemy in self.enemies:
            if enemy.pos.y > self.HEIGHT - self.DANGER_ZONE_HEIGHT:
                self.reward_buffer -= 10.0
                self.game_over = True
                return True
                
        if self.steps >= self.MAX_STEPS:
            # This is truncation, not termination
            return False
            
        return False
        
    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _render_game(self):
        # Draw walls
        pygame.draw.rect(self.screen, self.COLOR_WALLS, (0, 0, self.WIDTH, self.HEIGHT), 5)
        
        # Draw danger zone
        danger_surface = pygame.Surface((self.WIDTH, self.DANGER_ZONE_HEIGHT), pygame.SRCALPHA)
        danger_surface.fill(self.COLOR_DANGER)
        self.screen.blit(danger_surface, (0, self.HEIGHT - self.DANGER_ZONE_HEIGHT))
        
        # Draw slingshot
        self._render_slingshot()
        
        # Draw bubbles
        for bubble in self.bubbles:
            self._render_bubble(bubble)
            
        # Draw enemies
        for enemy in self.enemies:
            self._render_enemy(enemy)
            
        # Draw particles
        for p in self.particles:
            alpha = int(255 * (p.lifetime / p.max_lifetime))
            color_with_alpha = p.color + (alpha,)
            pygame.gfxdraw.filled_circle(self.screen, int(p.pos.x), int(p.pos.y), int(p.size), color_with_alpha)

    def _render_slingshot(self):
        post1 = self.slingshot_pos + pygame.Vector2(-15, 0)
        post2 = self.slingshot_pos + pygame.Vector2(15, 0)
        pygame.draw.circle(self.screen, self.COLOR_PLAYER, (int(post1.x), int(post1.y)), 5)
        pygame.draw.circle(self.screen, self.COLOR_PLAYER, (int(post2.x), int(post2.y)), 5)
        
        # Draw aim preview bubble and bands
        radius = self.BUBBLE_SIZES[self.current_bubble_size_idx]
        preview_pos = self.slingshot_pos + pygame.Vector2(math.cos(self.aim_angle), math.sin(self.aim_angle)) * (radius + 15)
        
        pygame.draw.line(self.screen, self.COLOR_PLAYER, post1, preview_pos, 2)
        pygame.draw.line(self.screen, self.COLOR_PLAYER, post2, preview_pos, 2)
        
        preview_color = self.BUBBLE_COLORS[self.current_bubble_size_idx % len(self.BUBBLE_COLORS)]
        pygame.gfxdraw.filled_circle(self.screen, int(preview_pos.x), int(preview_pos.y), radius, preview_color + (150,))
        pygame.gfxdraw.aacircle(self.screen, int(preview_pos.x), int(preview_pos.y), radius, preview_color + (200,))

    def _render_bubble(self, bubble):
        # Use gfxdraw for anti-aliased circles
        pos = (int(bubble.pos.x), int(bubble.pos.y))
        radius = int(bubble.radius)
        
        # Transparent fill
        surface = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
        pygame.draw.circle(surface, bubble.color + (100,), (radius, radius), radius)
        self.screen.blit(surface, (pos[0] - radius, pos[1] - radius))
        
        # Opaque border
        pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], radius, bubble.color + (200,))
        
        # Highlight/sheen
        sheen_pos = (int(pos[0] - radius * 0.4), int(pos[1] - radius * 0.4))
        pygame.gfxdraw.filled_circle(self.screen, sheen_pos[0], sheen_pos[1], int(radius * 0.2), (255, 255, 255, 128))

    def _render_enemy(self, enemy):
        pos = (int(enemy.pos.x), int(enemy.pos.y))
        radius = int(enemy.radius)
        color = self.COLOR_ENEMY
        
        if enemy.state == "trapped":
            # Pulsate when trapped
            alpha = 128 + 127 * math.sin(self.steps * 0.2)
            color = color + (int(alpha),)
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], radius, color)
        else:
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], radius, color)
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], radius, (0,0,0)) # Outline

    def _render_ui(self):
        # Level and Score
        level_text = self.font_small.render(f"Level: {self.level}", True, self.COLOR_TEXT)
        score_text = self.font_small.render(f"Score: {self.score:.2f}", True, self.COLOR_TEXT)
        self.screen.blit(level_text, (10, 10))
        self.screen.blit(score_text, (self.WIDTH - score_text.get_width() - 10, 10))
        
        # Remaining bubbles
        bubble_icon_radius = 8
        for i in range(self.bubbles_left):
            x = self.slingshot_pos.x + 40 + i * (bubble_icon_radius * 2.5)
            y = self.slingshot_pos.y
            if x > self.WIDTH - 20: break # Don't draw off-screen
            pygame.gfxdraw.filled_circle(self.screen, int(x), int(y), bubble_icon_radius, self.BUBBLE_COLORS[0])
            pygame.gfxdraw.aacircle(self.screen, int(x), int(y), bubble_icon_radius, (255,255,255))
            
        if self.game_over:
            outcome_text = "LEVEL CLEARED" if not self.enemies else "GAME OVER"
            text_surface = self.font_large.render(outcome_text, True, self.COLOR_TEXT)
            text_rect = text_surface.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
            self.screen.blit(text_surface, text_rect)
            
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "level": self.level,
            "bubbles_left": self.bubbles_left,
            "enemies_left": len(self.enemies),
        }

    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # This block allows for manual testing of the environment
    # It will not run in the evaluation environment, which uses a dummy video driver.
    # To run this, you may need to comment out the os.environ line at the top.
    os.environ["SDL_VIDEODRIVER"] = "x11" # Or "windows", "mac", etc.
    
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((GameEnv.WIDTH, GameEnv.HEIGHT))
    pygame.display.set_caption("Bubble Sling Test")
    clock = pygame.time.Clock()
    
    terminated = False
    
    while not terminated:
        # Default action is no-op
        movement = 0
        space_held = 0
        shift_held = 0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        if keys[pygame.K_SPACE]: space_held = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift_held = 1
        if keys[pygame.K_r]: # Reset key
            obs, info = env.reset()
            continue

        action = [movement, space_held, shift_held]
        obs, reward, terminated, truncated, info = env.step(action)
        
        if terminated or truncated:
            print(f"Game Over. Final Info: {info}")
            obs, info = env.reset()

        # Render the observation to the screen
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(30) # Limit to 30 FPS for playability

    env.close()