
# Generated: 2025-08-28T04:06:15.355749
# Source Brief: brief_02211.md
# Brief Index: 2211

        
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


# Helper classes for game objects
class Player:
    """Represents the player's hopper."""
    def __init__(self, pos, size):
        self.pos = pygame.Vector2(pos)
        self.vel = pygame.Vector2(0, 0)
        self.size = size
        self.color = (255, 50, 50)  # Bright Red
        self.on_ground = False
        self.rect = pygame.Rect(0, 0, self.size, self.size)
        
        self.jump_power = 10.0
        self.jump_angle = 0.0  # Degrees, 0 is straight up

        self.squash = 1.0
        self.rotation = 0.0

    def update(self, gravity, air_control_force):
        if not self.on_ground:
            self.vel += gravity
            self.pos += self.vel
            self.rotation = self.vel.angle_to(pygame.Vector2(1, 0))
            self.vel.x = max(-7, min(7, self.vel.x)) # Clamp horizontal speed

        if not self.on_ground and air_control_force != 0:
            self.vel.x += air_control_force
        
        self.squash += (1.0 - self.squash) * 0.2
        self.rect.center = (int(self.pos.x), int(self.pos.y - self.size / 2))

    def jump(self):
        if self.on_ground:
            # sfx: JumpCharge.wav -> Jump.wav
            angle_rad = math.radians(90 + self.jump_angle)
            self.vel.x = self.jump_power * math.cos(angle_rad)
            self.vel.y = -self.jump_power * math.sin(angle_rad)
            self.on_ground = False
            self.squash = 1.8  # Stretch for jump
            return True
        return False

    def land(self, platform_y):
        # sfx: Land.wav
        self.on_ground = True
        self.pos.y = platform_y
        self.vel.y = 0
        self.vel.x *= 0.5  # Friction
        self.squash = 0.5  # Squash on land

    def draw(self, surface):
        w = self.size * self.squash
        h = self.size / self.squash
        
        sprite = pygame.Surface((w, h), pygame.SRCALPHA)
        pygame.draw.ellipse(sprite, self.color, (0, 0, w, h))
        pygame.draw.ellipse(sprite, (255, 150, 150), (w * 0.2, h * 0.1, w * 0.6, h * 0.3))
        
        rot_angle = self.rotation if not self.on_ground else 0
        rotated_sprite = pygame.transform.rotate(sprite, rot_angle)
        
        new_rect = rotated_sprite.get_rect(center=(int(self.pos.x), int(self.pos.y - h / 2)))
        surface.blit(rotated_sprite, new_rect)

class Platform:
    """Represents a static platform."""
    def __init__(self, x, y, w, h, is_goal=False):
        self.rect = pygame.Rect(x, y, w, h)
        self.is_goal = is_goal
        self.color = (150, 255, 150) if is_goal else (100, 100, 120)
        self.outline_color = (200, 255, 200) if is_goal else (150, 150, 170)

    def draw(self, surface):
        pygame.draw.rect(surface, self.color, self.rect, border_radius=3)
        pygame.draw.rect(surface, self.outline_color, self.rect, width=2, border_radius=3)

class Coin:
    """Represents a collectable coin."""
    def __init__(self, x, y):
        self.pos = pygame.Vector2(x, y)
        self.base_y = y
        self.size = 8
        self.rect = pygame.Rect(x - self.size, y - self.size, self.size * 2, self.size * 2)
        self.collected = False
        self.anim_offset = random.uniform(0, math.pi * 2)

    def update(self, step):
        self.pos.y = self.base_y + math.sin(step * 0.1 + self.anim_offset) * 3
        self.rect.center = self.pos

    def draw(self, surface, step):
        if not self.collected:
            pulse = (math.sin(step * 0.2 + self.anim_offset) + 1) / 2
            current_size = self.size + pulse * 2
            color = (255, 215, 0)
            
            pygame.gfxdraw.filled_circle(surface, int(self.pos.x), int(self.pos.y), int(current_size), color)
            pygame.gfxdraw.aacircle(surface, int(self.pos.x), int(self.pos.y), int(current_size), color)
            pygame.gfxdraw.filled_circle(surface, int(self.pos.x), int(self.pos.y), int(current_size * 0.5), (255, 255, 150))

class Particle:
    """Represents a single particle for visual effects."""
    def __init__(self, pos, vel, size, lifespan, color):
        self.pos = pygame.Vector2(pos)
        self.vel = pygame.Vector2(vel)
        self.size = size
        self.lifespan = lifespan
        self.max_lifespan = lifespan
        self.color = color

    def update(self):
        self.pos += self.vel
        self.lifespan -= 1
        self.size = max(0, self.size - 0.1)
    
    def draw(self, surface):
        if self.lifespan > 0:
            alpha = max(0, min(255, int(255 * (self.lifespan / self.max_lifespan))))
            temp_surface = pygame.Surface((self.size*2, self.size*2), pygame.SRCALPHA)
            pygame.draw.circle(temp_surface, self.color + (alpha,), (int(self.size), int(self.size)), int(self.size))
            surface.blit(temp_surface, (int(self.pos.x - self.size), int(self.pos.y - self.size)), special_flags=pygame.BLEND_RGBA_ADD)


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: On a platform, use ←→ to aim and ↑↓ to set power. Press space to jump. In the air, use ←→ for air control."
    )

    game_description = (
        "A retro arcade platformer. Control a space hopper, jumping between platforms to reach the goal at the top. Collect coins for a higher score!"
    )

    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.WIDTH, self.HEIGHT = 640, 400
        
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        
        self.font = pygame.font.SysFont("monospace", 20, bold=True)
        
        # Colors & Game constants
        self.COLOR_BG = (15, 20, 40)
        self.COLOR_STAR = (180, 180, 200)
        self.GRAVITY = pygame.Vector2(0, 0.4)
        self.MAX_STEPS = 2000
        self.AIR_CONTROL = 0.2
        
        # State variables
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.player = None
        self.platforms = []
        self.coins = []
        self.particles = []
        self.stars = []

        self.reset()
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        
        # Procedurally generate level layout
        self.platforms = [Platform(self.WIDTH/2 - 50, self.HEIGHT - 40, 100, 15)] # Start
        self.platforms.append(Platform(self.WIDTH/2 - 40, 40, 80, 20, is_goal=True)) # Goal
        y_pos = self.HEIGHT - 120
        for i in range(8):
            px = self.np_random.integers(50, self.WIDTH - 120)
            py = y_pos - i * 45 + self.np_random.integers(-5, 5)
            pw = self.np_random.integers(60, 90)
            self.platforms.append(Platform(px, py, pw, 15))

        start_platform = self.platforms[0]
        self.player = Player(pos=(start_platform.rect.centerx, start_platform.rect.top), size=20)
        self.player.land(start_platform.rect.top)
        
        self.coins = []
        for p in self.platforms[2:]:
            if self.np_random.random() < 0.7:
                cx = p.rect.centerx + self.np_random.integers(-40, 40)
                cy = p.rect.top - self.np_random.integers(30, 70)
                self.coins.append(Coin(cx, cy))
        
        if not self.stars:
            for _ in range(100):
                self.stars.append((random.randint(0, self.WIDTH), random.randint(0, self.HEIGHT), random.randint(1, 2)))

        self.particles = []
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement, space_held, _ = action[0], action[1] == 1, action[2] == 1
        reward = 0
        
        # 1. Handle Input
        air_control_force = 0
        if self.player.on_ground:
            if movement == 1: self.player.jump_power = min(15, self.player.jump_power + 0.2)
            if movement == 2: self.player.jump_power = max(5, self.player.jump_power - 0.2)
            if movement == 3: self.player.jump_angle = max(-75, self.player.jump_angle - 2)
            if movement == 4: self.player.jump_angle = min(75, self.player.jump_angle + 2)
            if space_held and self.player.jump():
                self._create_particles(20, self.player.pos, 'jump')
        else:
            if movement == 3: air_control_force = -self.AIR_CONTROL
            if movement == 4: air_control_force = self.AIR_CONTROL

        # 2. Update State
        self.player.update(self.GRAVITY, air_control_force)
        for p in self.particles: p.update()
        self.particles = [p for p in self.particles if p.lifespan > 0]
        for c in self.coins: c.update(self.steps)

        # 3. Handle Collisions & Events
        if self.player.vel.y > 0:
            for p in self.platforms:
                if self.player.rect.colliderect(p.rect) and self.player.pos.y - self.player.vel.y <= p.rect.top:
                    self.player.land(p.rect.top)
                    self._create_particles(10, self.player.pos, 'land')
                    if p.is_goal:
                        self.game_over = True
                        reward += 100
                        # sfx: Win.wav
                    break
        
        for coin in self.coins:
            if not coin.collected and self.player.rect.colliderect(coin.rect):
                coin.collected = True
                self.score += 5
                reward += 5
                self._create_particles(15, coin.pos, 'coin')
                # sfx: Coin.wav
        
        # 4. Calculate Rewards
        reward += -self.player.vel.y * 0.02 # Rewards upward movement, penalizes downward
        
        # 5. Check Termination
        terminated = self.game_over
        if self.player.pos.y > self.HEIGHT + self.player.size:
            terminated = True
            reward -= 100
            # sfx: Lose.wav
        
        self.steps += 1
        if self.steps >= self.MAX_STEPS:
            terminated = True
        self.game_over = terminated
        
        return self._get_observation(), reward, terminated, False, self._get_info()
    
    def _create_particles(self, count, pos, p_type):
        for _ in range(count):
            if p_type == 'jump':
                angle = random.uniform(math.pi, math.pi * 2)
                speed = random.uniform(1, 4)
                vel = pygame.Vector2(math.cos(angle), math.sin(angle)) * speed
                color = random.choice([(200,200,200), (255,255,255)])
                lifespan = random.randint(20, 40)
            elif p_type == 'land':
                angle = random.uniform(0, math.pi * 2)
                speed = random.uniform(0.5, 2)
                vel = pygame.Vector2(math.cos(angle), math.sin(angle)) * speed
                color = (180, 180, 190)
                lifespan = random.randint(10, 30)
            elif p_type == 'coin':
                angle = random.uniform(0, math.pi * 2)
                speed = random.uniform(1, 5)
                vel = pygame.Vector2(math.cos(angle), math.sin(angle)) * speed
                color = random.choice([(255,215,0), (255,255,100)])
                lifespan = random.randint(20, 50)
            
            size = random.uniform(2, 5)
            self.particles.append(Particle(pos, vel, size, lifespan, color))

    def _render_game(self):
        for x, y, size in self.stars:
            pygame.draw.rect(self.screen, self.COLOR_STAR, (x, y, size, size))
        
        if self.player.on_ground: self._draw_trajectory()

        for p in self.platforms: p.draw(self.screen)
        for c in self.coins: c.draw(self.screen, self.steps)
        
        particle_surface = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
        for p in self.particles: p.draw(particle_surface)
        self.screen.blit(particle_surface, (0,0))
        
        self.player.draw(self.screen)
    
    def _draw_trajectory(self):
        pos = pygame.Vector2(self.player.pos)
        angle_rad = math.radians(90 + self.player.jump_angle)
        vel = pygame.Vector2(self.player.jump_power * math.cos(angle_rad), -self.player.jump_power * math.sin(angle_rad))
        
        for i in range(30):
            vel += self.GRAVITY
            pos += vel
            if i % 3 == 0:
                pygame.gfxdraw.filled_circle(self.screen, int(pos.x), int(pos.y), 2, (255, 255, 255, 100))

    def _render_ui(self):
        score_text = self.font.render(f"SCORE: {self.score}", True, (255, 255, 255))
        self.screen.blit(score_text, (10, 10))
        
        steps_text = self.font.render(f"STEPS: {self.steps}/{self.MAX_STEPS}", True, (255, 255, 255))
        self.screen.blit(steps_text, (self.WIDTH - steps_text.get_width() - 10, 10))

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _get_info(self):
        return {"score": self.score, "steps": self.steps}
        
    def close(self):
        pygame.quit()
        
    def validate_implementation(self):
        assert self.action_space.shape == (3,) and self.action_space.nvec.tolist() == [5, 2, 2]
        test_obs = self._get_observation()
        assert test_obs.shape == (self.HEIGHT, self.WIDTH, 3) and test_obs.dtype == np.uint8
        obs, info = self.reset()
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3) and isinstance(info, dict)
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(reward, (int, float)) and isinstance(term, bool) and not trunc
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    import time
    GameEnv.metadata["render_modes"].append("human")
    
    original_get_obs = GameEnv._get_observation
    def human_get_observation(self):
        if not hasattr(self, 'display'):
            pygame.display.init()
            self.display = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
        
        original_get_obs(self) # Call the original rendering logic
        self.display.blit(self.screen, (0, 0))
        pygame.display.flip()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    GameEnv._get_observation = human_get_observation

    env = GameEnv(render_mode="human")
    obs, info = env.reset()
    
    keys = {"up": False, "down": False, "left": False, "right": False, "space": False, "shift": False}
    print(f"\n{'='*30}\n{env.game_description}\n{env.user_guide}\n{'='*30}\n")

    terminated = False
    while not terminated:
        action = [0, 0, 0]
        for event in pygame.event.get():
            if event.type == pygame.QUIT: terminated = True
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r: obs, info = env.reset()
                if event.key in [pygame.K_UP, pygame.K_w]: keys["up"] = True
                if event.key in [pygame.K_DOWN, pygame.K_s]: keys["down"] = True
                if event.key in [pygame.K_LEFT, pygame.K_a]: keys["left"] = True
                if event.key in [pygame.K_RIGHT, pygame.K_d]: keys["right"] = True
                if event.key == pygame.K_SPACE: keys["space"] = True
                if event.key in [pygame.K_LSHIFT, pygame.K_RSHIFT]: keys["shift"] = True
            if event.type == pygame.KEYUP:
                if event.key in [pygame.K_UP, pygame.K_w]: keys["up"] = False
                if event.key in [pygame.K_DOWN, pygame.K_s]: keys["down"] = False
                if event.key in [pygame.K_LEFT, pygame.K_a]: keys["left"] = False
                if event.key in [pygame.K_RIGHT, pygame.K_d]: keys["right"] = False
                if event.key == pygame.K_SPACE: keys["space"] = False
                if event.key in [pygame.K_LSHIFT, pygame.K_RSHIFT]: keys["shift"] = False

        if keys["up"]: action[0] = 1
        elif keys["down"]: action[0] = 2
        elif keys["left"]: action[0] = 3
        elif keys["right"]: action[0] = 4
        if keys["space"]: action[1] = 1
        if keys["shift"]: action[2] = 1
        
        obs, reward, term, trunc, info = env.step(action)
        terminated = term
        
        if terminated:
            print(f"Episode finished. Score: {info['score']}, Steps: {info['steps']}")
            time.sleep(2)
            obs, info = env.reset()
            terminated = False

    env.close()