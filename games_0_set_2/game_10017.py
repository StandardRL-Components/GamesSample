import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T09:41:27.625054
# Source Brief: brief_00017.md
# Brief Index: 17
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

# Helper classes for game objects
class Barrel:
    def __init__(self, pos, speed_multiplier):
        self.pos = pygame.Vector2(pos)
        self.radius = 10
        self.speed = 2.5 * speed_multiplier
        self.vel = pygame.Vector2(self.speed, 0)
        self.rotation = 0
        self.history = []

    def update(self, platforms, wind_strength):
        self.history.append(self.pos.copy())
        if len(self.history) > 5:
            self.history.pop(0)

        on_platform = False
        target_y = 1000

        for p in platforms:
            if p.left <= self.pos.x <= p.right:
                slope = math.tan(math.radians(p.angle))
                plat_y_at_barrel_x = p.y + (self.pos.x - p.x) * slope
                if self.pos.y - self.radius < plat_y_at_barrel_x < self.pos.y + self.radius * 2:
                    target_y = min(target_y, plat_y_at_barrel_x)
                    on_platform = True
                    self.vel.x = self.speed if p.angle < 0 else -self.speed

        if on_platform:
            self.pos.y = target_y - self.radius
            self.vel.y = 0
        else:
            self.vel.y += 0.3 # Gravity

        self.vel.x += wind_strength * 0.1
        self.pos += self.vel
        self.rotation += self.vel.x * 5
        
    def draw(self, surface):
        # Motion blur
        for i, pos in enumerate(self.history):
            alpha = int(50 * (i / len(self.history)))
            if alpha > 0:
                self._draw_single(surface, pos, alpha)

        # Main barrel
        self._draw_single(surface, self.pos, 255)

    def _draw_single(self, surface, pos, alpha):
        color_main = (200, 50, 50, alpha)
        color_highlight = (230, 100, 100, alpha)
        
        pygame.gfxdraw.filled_circle(surface, int(pos.x), int(pos.y), self.radius, color_main)
        pygame.gfxdraw.aacircle(surface, int(pos.x), int(pos.y), self.radius, color_main)

        # Draw stripe for rotation
        angle_rad = math.radians(self.rotation)
        start = (pos.x - math.cos(angle_rad) * self.radius, pos.y - math.sin(angle_rad) * self.radius)
        end = (pos.x + math.cos(angle_rad) * self.radius, pos.y + math.sin(angle_rad) * self.radius)
        pygame.draw.line(surface, color_highlight, (int(start[0]), int(start[1])), (int(end[0]), int(end[1])), 3)


class Particle:
    def __init__(self, pos, vel, lifetime, start_color, end_color, start_size, end_size=0):
        self.pos = pygame.Vector2(pos)
        self.vel = pygame.Vector2(vel)
        self.lifetime = lifetime
        self.max_lifetime = lifetime
        self.start_color = start_color
        self.end_color = end_color
        self.start_size = start_size
        self.end_size = end_size

    def update(self):
        self.lifetime -= 1
        self.pos += self.vel
        self.vel *= 0.98

    def draw(self, surface):
        progress = self.lifetime / self.max_lifetime
        if progress < 0: return
        
        current_color = [
            int(self.end_color[i] + (self.start_color[i] - self.end_color[i]) * progress) for i in range(3)
        ]
        current_size = int(self.end_size + (self.start_size - self.end_size) * progress)

        if current_size > 0:
            pygame.gfxdraw.filled_circle(surface, int(self.pos.x), int(self.pos.y), current_size, current_color)
            pygame.gfxdraw.aacircle(surface, int(self.pos.x), int(self.pos.y), current_size, current_color)

class Platform:
    def __init__(self, x, y, w, h, angle=0):
        self.rect = pygame.Rect(x, y, w, h)
        self.angle = angle
        # For collision detection with player
        self.x, self.y, self.w, self.h = x, y, w, h
        self.left, self.right = x, x + w

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Climb a perilous structure of platforms and ladders to reach the top. "
        "Avoid rolling barrels and use your charged jump to navigate the ascent."
    )
    user_guide = (
        "Controls: Use ←→ arrow keys to run and ↑↓ to climb ladders. "
        "Press ↑ or release space to jump; hold space to charge your jump."
    )
    auto_advance = True

    # --- CONSTANTS ---
    WIDTH, HEIGHT = 640, 400
    COLOR_BG = (20, 30, 40)
    COLOR_SCAFFOLD = (101, 67, 33)
    COLOR_SCAFFOLD_DARK = (81, 53, 26)
    COLOR_LADDER = (150, 120, 90)
    COLOR_PLAYER = (50, 150, 255)
    COLOR_PLAYER_GLOW = (50, 150, 255)
    COLOR_PAULINE = (255, 105, 180)
    COLOR_DK = (139, 69, 19)
    COLOR_TEXT = (240, 240, 240)
    
    GRAVITY = 0.4
    PLAYER_ACCEL = 0.8
    PLAYER_FRICTION = 0.85
    PLAYER_CLIMB_SPEED = 2.0
    PLAYER_JUMP_BASE = -8.0
    JUMP_CHARGE_RATE = 0.2
    MAX_JUMP_CHARGE = 10
    
    GRIP_MAX = 100.0
    GRIP_DEPLETE_RATE = 0.5
    GRIP_REGEN_RATE = 2.0

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.observation_space = Box(low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8)
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("consolas", 24)
        
        self._setup_level()
        self.reset()

        self.validate_implementation()

    def _setup_level(self):
        self.platforms = [
            Platform(0, 370, 640, 30, 0),
            Platform(100, 300, 540, 10, -5),
            Platform(0, 220, 540, 10, 5),
            Platform(100, 140, 540, 10, -5),
        ]
        self.ladders = [
            pygame.Rect(80, 300, 20, 70),
            pygame.Rect(550, 220, 20, 80),
            pygame.Rect(80, 140, 20, 80),
        ]
        self.pauline_pos = pygame.Vector2(320, 105)
        self.dk_pos = pygame.Vector2(70, 110)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.player_pos = pygame.Vector2(100, 350)
        self.player_vel = pygame.Vector2(0, 0)
        self.player_size = 12
        self.is_grounded = False
        self.on_ladder = False
        
        self.jump_charge = 0
        self.space_was_held = False
        
        self.grip = self.GRIP_MAX
        self.last_y_level = self.player_pos.y // 80 # For ladder reward

        self.barrels = []
        self.particles = []
        
        self.steps = 0
        self.score = 0
        self.reward_this_step = 0
        self.game_over = False
        
        self.barrel_spawn_timer = 90
        self.barrel_speed_multiplier = 1.0
        self.max_barrels = 3
        
        self.wind_strength = 0
        self.wind_timer = 50
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        self.steps += 1
        self.reward_this_step = 0.01 # Survival reward

        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1

        self._update_difficulty()
        self._update_wind()
        
        self._handle_player_movement(movement, space_held)
        self._update_player_physics()
        
        self._update_barrels()
        self._update_particles()
        
        terminated = self._check_termination_conditions()
        
        self.score += self.reward_this_step
        
        return (
            self._get_observation(),
            self.reward_this_step,
            terminated,
            False,
            self._get_info()
        )

    def _update_difficulty(self):
        if self.steps > 0 and self.steps % 200 == 0:
            self.barrel_speed_multiplier = min(2.5, self.barrel_speed_multiplier + 0.1)
            self.max_barrels = min(8, self.max_barrels + 1)
            # 'Barrel frequency increase' SFX
    
    def _update_wind(self):
        self.wind_timer -= 1
        if self.wind_timer <= 0:
            self.wind_strength = self.np_random.uniform(-0.8, 0.8)
            self.wind_timer = self.np_random.integers(40, 60)
            # 'Wind gust' SFX
        
        if abs(self.wind_strength) > 0.1:
            for _ in range(2):
                p_pos = (0 if self.wind_strength > 0 else self.WIDTH, self.np_random.uniform(0, self.HEIGHT))
                p_vel = (self.wind_strength * self.np_random.uniform(3, 6), self.np_random.uniform(-0.5, 0.5))
                self.particles.append(Particle(p_pos, p_vel, 100, (200, 200, 220), (50, 50, 60), self.np_random.integers(1, 3)))

    def _handle_player_movement(self, movement, space_held):
        player_rect = pygame.Rect(self.player_pos.x - self.player_size, self.player_pos.y - self.player_size, self.player_size * 2, self.player_size * 2)
        can_climb = any(player_rect.colliderect(ladder) for ladder in self.ladders)
        
        # Climbing logic
        if can_climb and (movement == 1 or movement == 2):
            self.on_ladder = True
            self.is_grounded = True # To allow jumping off ladder
            self.player_vel.y = 0
            if movement == 1: self.player_pos.y -= self.PLAYER_CLIMB_SPEED # Up
            if movement == 2: self.player_pos.y += self.PLAYER_CLIMB_SPEED # Down
        else:
            self.on_ladder = False

        # Horizontal movement
        if not self.on_ladder:
            if movement == 3: self.player_vel.x -= self.PLAYER_ACCEL # Left
            if movement == 4: self.player_vel.x += self.PLAYER_ACCEL # Right

        # Jump logic
        if self.is_grounded:
            if space_held:
                self.jump_charge = min(self.MAX_JUMP_CHARGE, self.jump_charge + self.JUMP_CHARGE_RATE)
                # 'Jump charge' SFX
            
            # Jump on release or if 'up' is pressed
            if (not space_held and self.space_was_held) or (movement == 1 and not can_climb):
                jump_power = self.PLAYER_JUMP_BASE - self.jump_charge
                self.player_vel.y = jump_power
                self.is_grounded = False
                self.jump_charge = 0
                # 'Jump' SFX
                for _ in range(15):
                    angle = self.np_random.uniform(math.pi, 2 * math.pi)
                    speed = self.np_random.uniform(1, 3)
                    vel = (math.cos(angle) * speed, math.sin(angle) * speed)
                    self.particles.append(Particle(self.player_pos, vel, 20, (200, 200, 255), self.COLOR_PLAYER, 3))
        
        self.space_was_held = space_held
    
    def _update_player_physics(self):
        if not self.on_ladder:
            self.player_vel.y += self.GRAVITY
        
        self.player_vel.x *= self.PLAYER_FRICTION
        if abs(self.player_vel.x) < 0.1: self.player_vel.x = 0

        self.player_pos += self.player_vel
        
        # Collision with platforms
        self.is_grounded = False
        if not self.on_ladder:
            for p in self.platforms:
                is_above = self.player_pos.y - self.player_size < p.rect.top
                is_falling = self.player_vel.y > 0
                is_horizontally_aligned = p.rect.left < self.player_pos.x < p.rect.right
                
                if is_falling and is_above and is_horizontally_aligned and self.player_pos.y > p.rect.top - self.player_vel.y:
                    self.player_pos.y = p.rect.top - self.player_size
                    self.player_vel.y = 0
                    self.is_grounded = True
                    # 'Land' SFX
                    break
        
        # Grip strength
        if not self.is_grounded and not self.on_ladder:
            self.grip = max(0, self.grip - self.GRIP_DEPLETE_RATE)
            self.reward_this_step -= 0.01 # Small penalty for using grip
            if self.grip == 0:
                self.player_vel.y = max(self.player_vel.y, 1.0) # Force downwards
        else:
            self.grip = min(self.GRIP_MAX, self.grip + self.GRIP_REGEN_RATE)
            
        # Ladder reward
        current_y_level = self.player_pos.y // 80
        if self.on_ladder and current_y_level < self.last_y_level:
            self.reward_this_step += 1.0
            self.last_y_level = current_y_level
        elif not self.on_ladder:
            self.last_y_level = current_y_level


        # Keep player in bounds
        self.player_pos.x = np.clip(self.player_pos.x, self.player_size, self.WIDTH - self.player_size)
    
    def _update_barrels(self):
        self.barrel_spawn_timer -= 1
        if self.barrel_spawn_timer <= 0 and len(self.barrels) < self.max_barrels:
            self.barrels.append(Barrel(self.dk_pos, self.barrel_speed_multiplier))
            self.barrel_spawn_timer = self.np_random.integers(90, 120) // (self.barrel_speed_multiplier)
            # 'Barrel spawn' SFX

        for barrel in self.barrels[:]:
            barrel.update(self.platforms, self.wind_strength)
            if barrel.pos.y > self.HEIGHT + 50 or barrel.pos.x < -50 or barrel.pos.x > self.WIDTH + 50:
                self.barrels.remove(barrel)
    
    def _update_particles(self):
        for p in self.particles[:]:
            p.update()
            if p.lifetime <= 0:
                self.particles.remove(p)

    def _check_termination_conditions(self):
        # Hit by barrel
        for barrel in self.barrels:
            if self.player_pos.distance_to(barrel.pos) < self.player_size + barrel.radius:
                self.reward_this_step = -50
                self.game_over = True
                # 'Player hit' SFX
                return True
        
        # Fell off world
        if self.player_pos.y > self.HEIGHT + self.player_size:
            self.reward_this_step = -50
            self.game_over = True
            # 'Player fall' SFX
            return True
            
        # Reached Pauline
        if self.player_pos.distance_to(self.pauline_pos) < self.player_size + 15:
            self.reward_this_step = 100
            self.game_over = True
            # 'Victory' SFX
            return True

        # Max steps
        if self.steps >= 2000:
            self.game_over = True
            return True
            
        return False
        
    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "grip": self.grip}
        
    def _render_game(self):
        # Scaffolding and ladders
        for p in self.platforms:
            pygame.draw.rect(self.screen, self.COLOR_SCAFFOLD_DARK, p.rect.move(0, 5))
            pygame.draw.rect(self.screen, self.COLOR_SCAFFOLD, p.rect)
        for l in self.ladders:
            for i in range(l.height // 10):
                pygame.draw.rect(self.screen, self.COLOR_LADDER, (l.x, l.y + i * 10, l.width, 3))

        # Particles
        for p in self.particles:
            p.draw(self.screen)

        # Pauline
        pygame.gfxdraw.filled_circle(self.screen, int(self.pauline_pos.x), int(self.pauline_pos.y), 15, self.COLOR_PAULINE)
        pygame.gfxdraw.aacircle(self.screen, int(self.pauline_pos.x), int(self.pauline_pos.y), 15, self.COLOR_PAULINE)
        
        # DK
        pygame.gfxdraw.filled_circle(self.screen, int(self.dk_pos.x), int(self.dk_pos.y), 18, self.COLOR_DK)
        
        # Barrels
        for barrel in self.barrels:
            barrel.draw(self.screen)
        
        # Player
        self._draw_player()
        
    def _draw_player(self):
        pos_x, pos_y = int(self.player_pos.x), int(self.player_pos.y)
        
        # Glow
        glow_radius = int(self.player_size * (1.5 + self.jump_charge / self.MAX_JUMP_CHARGE * 0.8))
        glow_alpha = int(80 + self.jump_charge / self.MAX_JUMP_CHARGE * 100)
        s = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
        pygame.draw.circle(s, (*self.COLOR_PLAYER_GLOW, glow_alpha), (glow_radius, glow_radius), glow_radius)
        self.screen.blit(s, (pos_x - glow_radius, pos_y - glow_radius), special_flags=pygame.BLEND_RGBA_ADD)
        
        # Body
        body_squish = self.jump_charge / self.MAX_JUMP_CHARGE * 0.4
        body_height = self.player_size * (1 - body_squish)
        body_width = self.player_size * (1 + body_squish)
        
        pygame.gfxdraw.filled_circle(self.screen, pos_x, pos_y, int(body_width), self.COLOR_PLAYER)
        pygame.gfxdraw.aacircle(self.screen, pos_x, pos_y, int(body_width), self.COLOR_PLAYER)

    def _render_ui(self):
        # Score
        score_text = self.font.render(f"SCORE: {int(self.score):05d}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))

        # Grip bar
        if self.grip < self.GRIP_MAX:
            bar_width = 30
            bar_height = 5
            bar_x = self.player_pos.x - bar_width / 2
            bar_y = self.player_pos.y - self.player_size - 15
            
            grip_ratio = self.grip / self.GRIP_MAX
            
            # Color transition from green to red
            bar_color = (
                int(255 * (1 - grip_ratio)),
                int(200 * grip_ratio),
                50
            )

            pygame.draw.rect(self.screen, (0,0,0), (bar_x-1, bar_y-1, bar_width+2, bar_height+2))
            pygame.draw.rect(self.screen, bar_color, (bar_x, bar_y, bar_width * grip_ratio, bar_height))

    def close(self):
        pygame.quit()

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
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # --- Pygame setup for human play ---
    human_screen = pygame.display.set_mode((GameEnv.WIDTH, GameEnv.HEIGHT))
    pygame.display.set_caption("Physics Platformer")
    human_clock = pygame.time.Clock()
    
    terminated = False
    total_reward = 0
    
    # Action state
    movement = 0
    space_held = 0
    shift_held = 0

    print("\n--- Human Controls ---")
    print("Arrows: Move Left/Right, Climb Up/Down")
    print("Space: Hold to charge jump, release to jump")
    print("R: Reset environment")
    print("----------------------\n")

    while not terminated:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                total_reward = 0
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE: space_held = 1
                if event.key == pygame.K_LSHIFT: shift_held = 1
            if event.type == pygame.KEYUP:
                if event.key == pygame.K_SPACE: space_held = 0
                if event.key == pygame.K_LSHIFT: shift_held = 0

        keys = pygame.key.get_pressed()
        movement = 0 # No-op by default
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        action = [movement, space_held, shift_held]
        
        obs, reward, term, trunc, info = env.step(action)
        total_reward += reward
        
        if term:
            print(f"Episode finished. Total Reward: {total_reward:.2f}, Score: {info['score']:.2f}, Steps: {info['steps']}")
            obs, info = env.reset()
            total_reward = 0

        # Render the observation to the human-visible screen
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        human_screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        human_clock.tick(30) # Run at 30 FPS for human play
        
    env.close()