
# Generated: 2025-08-27T22:58:59.878381
# Source Brief: brief_03308.md
# Brief Index: 3308

        
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


# Helper class for particles to add visual flair
class Particle:
    def __init__(self, pos, vel, life, color_start, color_end, radius_start, radius_end):
        self.pos = list(pos)
        self.vel = list(vel)
        self.life = life
        self.max_life = life
        self.color_start = color_start
        self.color_end = color_end
        self.radius_start = radius_start
        self.radius_end = radius_end

    def update(self):
        self.pos[0] += self.vel[0]
        self.pos[1] += self.vel[1]
        self.life -= 1

    def draw(self, surface):
        if self.life > 0:
            life_ratio = self.life / self.max_life
            radius = int(self.radius_start * life_ratio + self.radius_end * (1 - life_ratio))
            color = [
                int(self.color_start[i] * life_ratio + self.color_end[i] * (1 - life_ratio))
                for i in range(3)
            ]
            if radius > 0:
                pygame.draw.circle(surface, color, (int(self.pos[0]), int(self.pos[1])), radius)


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Press SPACE to jump over low obstacles and hold SHIFT to slide under high ones."
    )

    game_description = (
        "A fast-paced side-scrolling runner. Jump and slide to avoid obstacles, "
        "collect time bonuses, and reach the finish line before time runs out."
    )

    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.WIDTH, self.HEIGHT = 640, 400
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.Font(None, 28)
        self.font_large = pygame.font.Font(None, 72)
        
        # --- Game Constants ---
        self.FPS = 30
        self.LEVEL_LENGTH = 7000
        self.INITIAL_TIME = 30 * self.FPS
        self.GROUND_Y = self.HEIGHT - 60

        # --- Colors ---
        self.COLOR_BG_TOP = (40, 20, 80)
        self.COLOR_BG_BOTTOM = (100, 60, 140)
        self.COLOR_GROUND = (30, 30, 40)
        self.COLOR_PLAYER = (255, 80, 80)
        self.COLOR_OBSTACLE = (120, 200, 255)
        self.COLOR_TIME_BONUS = (255, 215, 0)
        self.COLOR_TEXT = (240, 240, 240)
        self.COLOR_TEXT_SHADOW = (20, 20, 30)

        # --- Player Properties ---
        self.PLAYER_SCREEN_X = 120
        self.PLAYER_RUN_SIZE = (24, 48)
        self.PLAYER_SLIDE_SIZE = (48, 24)
        self.GRAVITY = -1.3
        self.JUMP_STRENGTH = 21
        self.INITIAL_RUN_SPEED = 8.0
        self.SLIDE_DURATION = 15

        # --- Game State (initialized in reset) ---
        self.obstacles = []
        self.time_bonuses = []
        self.particles = []
        self.np_random = None
        self.run_speed = 0
        self.player_world_x = 0
        self.player_y = 0
        self.player_vy = 0
        self.player_state = "RUNNING"
        self.slide_timer = 0
        self.time_remaining = 0
        self.next_obstacle_x = 0
        self.next_bonus_x = 0
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False
        self.last_reward_events = []
        
        # self.validate_implementation() # For self-checking during development

    def _get_player_rect(self):
        if self.player_state == "SLIDING":
            w, h = self.PLAYER_SLIDE_SIZE
        else:
            w, h = self.PLAYER_RUN_SIZE
        
        bob = 0
        if self.player_state == "RUNNING":
            bob = abs(math.sin(self.steps * 0.5) * 4)

        return pygame.Rect(self.PLAYER_SCREEN_X - w / 2, self.player_y - h + bob, w, h)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.run_speed = self.INITIAL_RUN_SPEED
        self.player_world_x = 0
        self.player_y = self.GROUND_Y
        self.player_vy = 0
        self.player_state = "RUNNING"
        self.slide_timer = 0
        
        self.obstacles.clear()
        self.time_bonuses.clear()
        self.particles.clear()
        
        self.time_remaining = self.INITIAL_TIME
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False

        self.next_obstacle_x = 400
        self.next_bonus_x = 600

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        self.clock.tick(self.FPS)
        self.steps += 1
        self.last_reward_events.clear()

        self._handle_input(action)
        self._update_player()
        self.player_world_x += self.run_speed

        self._update_entities()
        self._manage_spawns()
        self._update_particles()
        
        self.time_remaining -= 1
        if self.steps > 0 and self.steps % 500 == 0:
            self.run_speed += 0.2

        reward = self._calculate_reward()
        terminated = self._check_termination()
        
        self.score += reward

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
        is_on_ground = self.player_y == self.GROUND_Y

        if shift_held:
            if self.player_state != "SLIDING":
                self.player_state = "SLIDING"
                self.slide_timer = self.SLIDE_DURATION
                # sfx: slide_start
        elif space_held and is_on_ground:
            self.player_vy = self.JUMP_STRENGTH
            self.player_state = "JUMPING"
            # sfx: jump

    def _update_player(self):
        is_on_ground = self.player_y == self.GROUND_Y

        if self.player_state == "SLIDING":
            self.slide_timer -= 1
            if self.slide_timer <= 0:
                self.player_state = "RUNNING"
            if self.np_random.random() < 0.5:
                self._create_slide_particles()

        if not is_on_ground or self.player_state == "JUMPING":
            self.player_y -= self.player_vy
            self.player_vy += self.GRAVITY

        if self.player_y >= self.GROUND_Y:
            self.player_y = self.GROUND_Y
            self.player_vy = 0
            if self.player_state == "JUMPING":
                self.player_state = "RUNNING"
                # sfx: land

    def _update_entities(self):
        player_rect = self._get_player_rect()

        for obstacle in self.obstacles[:]:
            obstacle.x -= self.run_speed
            if obstacle.right < 0:
                self.obstacles.remove(obstacle)
            elif not self.game_over and obstacle.colliderect(player_rect):
                self.game_over = True
                self.last_reward_events.append("collision")
                # sfx: collision_hit
                self._create_explosion(player_rect.center, self.COLOR_PLAYER, 15)
                self._create_explosion(obstacle.center, self.COLOR_OBSTACLE, 25)

        for bonus in self.time_bonuses[:]:
            bonus.x -= self.run_speed
            if bonus.right < 0:
                self.time_bonuses.remove(bonus)
            elif bonus.colliderect(player_rect):
                self.time_bonuses.remove(bonus)
                self.time_remaining = min(self.INITIAL_TIME, self.time_remaining + 2 * self.FPS)
                self.last_reward_events.append("time_bonus")
                # sfx: collect_bonus
                self._create_explosion(bonus.center, self.COLOR_TIME_BONUS, 20)

    def _manage_spawns(self):
        if self.player_world_x > self.next_obstacle_x:
            spawn_x = self.WIDTH + 50
            is_high_obstacle = self.np_random.choice([True, False])
            if is_high_obstacle:
                obstacle_rect = pygame.Rect(spawn_x, self.GROUND_Y - 100, 50, 60)
            else:
                obstacle_rect = pygame.Rect(spawn_x, self.GROUND_Y - 30, 40, 30)
            self.obstacles.append(obstacle_rect)
            self.next_obstacle_x += self.np_random.integers(350, 550)

        if self.player_world_x > self.next_bonus_x:
            spawn_x = self.WIDTH + 100
            spawn_y = self.np_random.integers(self.GROUND_Y - 150, self.GROUND_Y - 50)
            bonus_rect = pygame.Rect(spawn_x, spawn_y, 20, 20)
            self.time_bonuses.append(bonus_rect)
            self.next_bonus_x += self.np_random.integers(800, 1200)

    def _update_particles(self):
        for p in self.particles[:]:
            p.update()
            if p.life <= 0:
                self.particles.remove(p)

    def _calculate_reward(self):
        reward = 0.1
        if "time_bonus" in self.last_reward_events:
            reward += 1.0
        if self._check_termination():
            if self.win:
                reward += 50.0
            elif "collision" in self.last_reward_events:
                reward += -50.0
            else:
                reward += -10.0
        return reward

    def _check_termination(self):
        if self.game_over: return True
        if self.time_remaining <= 0:
            self.game_over = True
            return True
        if self.player_world_x >= self.LEVEL_LENGTH:
            self.game_over = True
            self.win = True
            return True
        if self.steps >= 1000:
            self.game_over = True
            return True
        return False
        
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "time_remaining": self.time_remaining / self.FPS,
            "world_progress": self.player_world_x / self.LEVEL_LENGTH
        }

    def _get_observation(self):
        self._render_background()
        self._render_game_elements()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_text(self, text, pos, font, color, shadow_color=None):
        if shadow_color:
            text_surf_shadow = font.render(text, True, shadow_color)
            self.screen.blit(text_surf_shadow, (pos[0] + 2, pos[1] + 2))
        text_surf = font.render(text, True, color)
        self.screen.blit(text_surf, pos)

    def _render_background(self):
        for y in range(self.HEIGHT):
            ratio = y / self.HEIGHT
            color = [int(self.COLOR_BG_TOP[i] * (1 - ratio) + self.COLOR_BG_BOTTOM[i] * ratio) for i in range(3)]
            pygame.draw.line(self.screen, color, (0, y), (self.WIDTH, y))
        pygame.draw.rect(self.screen, self.COLOR_GROUND, (0, self.GROUND_Y, self.WIDTH, self.HEIGHT - self.GROUND_Y))

    def _render_game_elements(self):
        for p in self.particles:
            p.draw(self.screen)

        for bonus in self.time_bonuses:
            for i in range(5, 0, -1):
                alpha = 80 - i * 15
                color = (*self.COLOR_TIME_BONUS, alpha)
                s = pygame.Surface((bonus.width + i*4, bonus.height + i*4), pygame.SRCALPHA)
                pygame.draw.ellipse(s, color, s.get_rect())
                self.screen.blit(s, (bonus.x - i*2, bonus.y - i*2))
            pygame.draw.ellipse(self.screen, self.COLOR_TIME_BONUS, bonus)

        for obstacle in self.obstacles:
            pygame.draw.rect(self.screen, self.COLOR_OBSTACLE, obstacle, border_radius=3)
            highlight_rect = pygame.Rect(obstacle.x+2, obstacle.y+2, obstacle.width-4, 5)
            pygame.draw.rect(self.screen, (200, 230, 255), highlight_rect, border_radius=2)

        player_rect = self._get_player_rect()
        for i in range(10, 0, -2):
            alpha = 50 - i * 5
            color = (*self.COLOR_PLAYER, alpha)
            s = pygame.Surface((player_rect.width + i*2, player_rect.height + i*2), pygame.SRCALPHA)
            pygame.draw.rect(s, color, s.get_rect(), border_radius=8)
            self.screen.blit(s, (player_rect.x - i, player_rect.y - i))
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, player_rect, border_radius=5)

    def _render_ui(self):
        time_str = f"TIME: {max(0, self.time_remaining / self.FPS):.1f}"
        self._render_text(time_str, (self.WIDTH - 160, 20), self.font_small, self.COLOR_TEXT, self.COLOR_TEXT_SHADOW)

        progress_ratio = min(1.0, self.player_world_x / self.LEVEL_LENGTH)
        bar_width = self.WIDTH - 40
        bar_y = self.HEIGHT - 25
        pygame.draw.rect(self.screen, self.COLOR_TEXT_SHADOW, (20, bar_y, bar_width, 10), border_radius=5)
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, (20, bar_y, bar_width * progress_ratio, 10), border_radius=5)

        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 128))
            self.screen.blit(overlay, (0, 0))
            message = "YOU WIN!" if self.win else "GAME OVER"
            text_surf = self.font_large.render(message, True, self.COLOR_TEXT)
            text_rect = text_surf.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2))
            self.screen.blit(text_surf, text_rect)

    def _create_explosion(self, pos, color, num_particles):
        for _ in range(num_particles):
            angle = self.np_random.random() * 2 * math.pi
            speed = self.np_random.random() * 5 + 2
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            life = self.np_random.integers(15, 30)
            p = Particle(pos, vel, life, color, (20,20,30), 8, 1)
            self.particles.append(p)

    def _create_slide_particles(self):
        player_rect = self._get_player_rect()
        pos = (player_rect.left, player_rect.bottom - 5)
        vel = [-self.run_speed - 2, self.np_random.random() * 2 - 1]
        life = self.np_random.integers(10, 20)
        p = Particle(pos, vel, life, (200,200,220), (50,50,60), 4, 0)
        self.particles.append(p)
        
    def close(self):
        pygame.quit()

if __name__ == "__main__":
    env = GameEnv(render_mode="rgb_array")
    env.reset()
    
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Arcade Runner")
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    while running:
        keys = pygame.key.get_pressed()
        space_held = keys[pygame.K_SPACE]
        shift_held = keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]
        
        action = [0, 1 if space_held else 0, 1 if shift_held else 0]
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                print("Resetting game.")
                obs, info = env.reset()
                total_reward = 0
                
        if terminated or truncated:
            print(f"Game Over. Final Score: {info['score']:.2f}, Total Steps: {info['steps']}")
            pygame.time.wait(2000)
            obs, info = env.reset()
            total_reward = 0

        clock.tick(env.FPS)
        
    env.close()