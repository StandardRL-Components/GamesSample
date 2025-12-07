import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import os
import os
import pygame


# Set the SDL video driver to dummy to run Pygame headless
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: Use ←→ to aim your jump. Press space to jump. Hold shift while jumping for a power jump."
    )

    game_description = (
        "Hop between moving platforms to reach the goal at the top before time runs out. Be quick and precise!"
    )

    auto_advance = True

    # --- Constants ---
    WIDTH, HEIGHT = 640, 400
    FPS = 30
    MAX_EPISODE_STEPS = 1000
    TIME_LIMIT_SECONDS = 60

    # Colors
    COLOR_BG_TOP = (40, 50, 80)
    COLOR_BG_BOTTOM = (80, 100, 140)
    COLOR_PLAYER = (50, 255, 50)
    COLOR_PLAYER_GLOW = (150, 255, 150, 50)
    COLOR_PLATFORM = (240, 240, 240)
    COLOR_PLATFORM_SHADOW = (180, 180, 180)
    COLOR_GOAL = (255, 215, 0)
    COLOR_GOAL_SHADOW = (200, 160, 0)
    COLOR_TEXT = (255, 255, 255)
    COLOR_TEXT_SHADOW = (0, 0, 0)

    # Physics
    GRAVITY = 0.4
    FRICTION = 0.95
    JUMP_POWER_SHORT = 9.0
    JUMP_POWER_LONG = 12.0
    JUMP_HORIZONTAL_SPEED = 4.0

    # Player
    PLAYER_SIZE = (16, 16)

    # Platforms
    PLATFORM_COUNT = 15
    PLATFORM_SIZE = (80, 12)
    MAX_PLATFORM_SPEED = 1.0

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
        self.font_ui = pygame.font.SysFont("monospace", 20, bold=True)
        self.font_big = pygame.font.SysFont("monospace", 48, bold=True)

        self.player_pos = pygame.Vector2(0, 0)
        self.player_vel = pygame.Vector2(0, 0)
        self.platforms = []
        self.particles = []
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.timer = 0
        self.on_platform = False
        self.last_space_held = False
        self.max_height_reached = self.HEIGHT
        self.current_platform_index = -1
        self.platform_speed_multiplier = 1.0
        
        self.bg_surface = self._create_gradient_background()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.timer = self.TIME_LIMIT_SECONDS * self.FPS
        self.last_space_held = False
        self.max_height_reached = self.HEIGHT
        self.platform_speed_multiplier = 1.0
        
        self._generate_platforms()
        
        start_platform = self.platforms[0]['rect']
        self.player_pos = pygame.Vector2(start_platform.centerx, start_platform.top - self.PLAYER_SIZE[1])
        self.player_vel = pygame.Vector2(0, 0)
        self.on_platform = True
        self.current_platform_index = 0
        
        self.particles.clear()

        return self._get_observation(), self._get_info()

    def step(self, action):
        reward = 0
        terminated = False

        if not self.game_over:
            self._handle_input(action)
            self._update_player()
            self._update_platforms()
            reward += self._check_collisions()
            
            self.timer -= 1
            self.steps += 1
            self.platform_speed_multiplier = 1.0 + (self.steps // 100) * 0.05

        self._update_particles()
        
        termination_reward, terminated = self._check_termination()
        reward += termination_reward
        
        if terminated:
            self.game_over = True

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_input(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        jump_triggered = space_held and not self.last_space_held
        
        if self.on_platform and jump_triggered and movement != 0:
            jump_power = self.JUMP_POWER_LONG if shift_held else self.JUMP_POWER_SHORT
            
            if movement == 1: # Up
                self.player_vel.y = -jump_power
            elif movement == 2: # Down (drop through)
                self.player_pos.y += 5
            elif movement == 3: # Left
                self.player_vel.x = -self.JUMP_HORIZONTAL_SPEED
                self.player_vel.y = -jump_power * 0.8
            elif movement == 4: # Right
                self.player_vel.x = self.JUMP_HORIZONTAL_SPEED
                self.player_vel.y = -jump_power * 0.8
            
            self.on_platform = False
            self._create_particles(self.player_pos + pygame.Vector2(self.PLAYER_SIZE[0]/2, self.PLAYER_SIZE[1]), 15, self.COLOR_PLAYER, 'jump')

        self.last_space_held = space_held

    def _update_player(self):
        if not self.on_platform:
            self.player_vel.y += self.GRAVITY
            self.player_pos += self.player_vel
            self.player_vel.x *= self.FRICTION

        if self.player_pos.x < 0:
            self.player_pos.x = self.WIDTH
        elif self.player_pos.x > self.WIDTH:
            self.player_pos.x = 0

    def _update_platforms(self):
        for plat in self.platforms[1:]: # Don't move the start platform
            plat['rect'].y += plat['vy'] * self.platform_speed_multiplier
            if plat['rect'].top < plat['range'][0] or plat['rect'].bottom > plat['range'][1]:
                plat['vy'] *= -1

    def _update_particles(self):
        self.particles = [p for p in self.particles if p['life'] > 0]
        for p in self.particles:
            p['pos'] += p['vel']
            p['life'] -= 1
            p['vel'].y += 0.1 # particle gravity

    def _check_collisions(self):
        reward = 0
        player_rect = pygame.Rect(self.player_pos, self.PLAYER_SIZE)
        
        if self.player_vel.y > 0:
            for i, plat in enumerate(self.platforms):
                if player_rect.colliderect(plat['rect']):
                    if abs(player_rect.bottom - plat['rect'].top) < self.player_vel.y + 1 and player_rect.right > plat['rect'].left and player_rect.left < plat['rect'].right:
                        self.on_platform = True
                        self.player_vel = pygame.Vector2(0, 0)
                        self.player_pos.y = plat['rect'].top - self.PLAYER_SIZE[1]
                        
                        self._create_particles(pygame.Vector2(player_rect.midbottom), 10, self.COLOR_PLATFORM, 'land')

                        if i != self.current_platform_index:
                            reward += 1.0
                            
                            if self.current_platform_index != -1 and i == self.current_platform_index + 1:
                                reward -= 0.02
                            
                            self.current_platform_index = i
                            
                            if plat['rect'].centery < self.max_height_reached:
                                reward += 5.0
                                self.score += 50
                                self.max_height_reached = plat['rect'].centery
                        
                        self.score += 10
                        break
        
        if self.on_platform:
            reward += 0.1
            plat_rect = self.platforms[self.current_platform_index]['rect']
            self.player_pos.y = plat_rect.top - self.PLAYER_SIZE[1]

        return reward

    def _check_termination(self):
        if self.player_pos.y > self.HEIGHT:
            return -10.0, True
        
        if self.timer <= 0:
            return -50.0, True
            
        if self.current_platform_index == len(self.platforms) - 1:
            self.score += 1000
            return 100.0, True
        
        if self.steps >= self.MAX_EPISODE_STEPS:
            return 0.0, True

        return 0.0, False

    def _generate_platforms(self):
        self.platforms.clear()
        
        start_rect = pygame.Rect(0, 0, 120, 20)
        start_rect.centerx = self.WIDTH / 2
        start_rect.bottom = self.HEIGHT - 20
        self.platforms.append({'rect': start_rect, 'vy': 0, 'range': (0,0), 'is_goal': False})

        y_pos = start_rect.top
        last_x = start_rect.centerx
        
        for i in range(self.PLATFORM_COUNT - 2):
            y_spacing = self.np_random.uniform(60, 80)
            y_pos -= y_spacing
            
            x_offset = self.np_random.uniform(-120, 120)
            x_pos = last_x + x_offset
            x_pos = np.clip(x_pos, self.PLATFORM_SIZE[0]/2, self.WIDTH - self.PLATFORM_SIZE[0]/2)
            
            plat_rect = pygame.Rect(0, 0, *self.PLATFORM_SIZE)
            plat_rect.center = (x_pos, y_pos)
            
            vy = self.np_random.uniform(0.5, self.MAX_PLATFORM_SPEED) * self.np_random.choice([-1, 1])
            move_range = self.np_random.uniform(10, 30)
            
            self.platforms.append({
                'rect': plat_rect,
                'vy': vy,
                'range': (y_pos - move_range, y_pos + move_range),
                'is_goal': False
            })
            last_x = x_pos
            
        goal_rect = pygame.Rect(0, 0, self.WIDTH, 30)
        goal_rect.top = 0
        self.platforms.append({'rect': goal_rect, 'vy': 0, 'range': (0,0), 'is_goal': True})

    def _get_observation(self):
        self.screen.blit(self.bg_surface, (0, 0))
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _create_gradient_background(self):
        bg = pygame.Surface((self.WIDTH, self.HEIGHT))
        for y in range(self.HEIGHT):
            interp = y / self.HEIGHT
            color = (
                self.COLOR_BG_TOP[0] * (1 - interp) + self.COLOR_BG_BOTTOM[0] * interp,
                self.COLOR_BG_TOP[1] * (1 - interp) + self.COLOR_BG_BOTTOM[1] * interp,
                self.COLOR_BG_TOP[2] * (1 - interp) + self.COLOR_BG_BOTTOM[2] * interp,
            )
            pygame.draw.line(bg, color, (0, y), (self.WIDTH, y))
        return bg

    def _render_game(self):
        for plat in self.platforms:
            color = self.COLOR_GOAL if plat['is_goal'] else self.COLOR_PLATFORM
            shadow_color = self.COLOR_GOAL_SHADOW if plat['is_goal'] else self.COLOR_PLATFORM_SHADOW
            
            shadow_rect = plat['rect'].copy()
            shadow_rect.y += 4
            pygame.draw.rect(self.screen, shadow_color, shadow_rect, border_radius=3)
            pygame.draw.rect(self.screen, color, plat['rect'], border_radius=3)
            
        for p in self.particles:
            alpha = max(0, min(255, int(255 * (p['life'] / p['max_life']))))
            # Create a temporary color object to set alpha, as p['color'] might be shared
            color = p['color']
            temp_color = pygame.Color(color.r, color.g, color.b, alpha)
            pygame.draw.circle(self.screen, temp_color, p['pos'], int(p['life'] / 3))

        player_rect = pygame.Rect(self.player_pos, self.PLAYER_SIZE)
        
        glow_surface = pygame.Surface((self.PLAYER_SIZE[0]*2, self.PLAYER_SIZE[1]*2), pygame.SRCALPHA)
        pygame.draw.circle(glow_surface, self.COLOR_PLAYER_GLOW, glow_surface.get_rect().center, self.PLAYER_SIZE[0])
        self.screen.blit(glow_surface, (player_rect.centerx - self.PLAYER_SIZE[0], player_rect.centery - self.PLAYER_SIZE[1]))
        
        inner_rect = player_rect.inflate(-6, -6)
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, player_rect, border_radius=3)
        pygame.draw.rect(self.screen, (200, 255, 200), inner_rect, border_radius=2)
        
    def _render_ui(self):
        score_text = f"SCORE: {self.score}"
        self._draw_text(score_text, (15, 10), self.font_ui, self.COLOR_TEXT, self.COLOR_TEXT_SHADOW)
        
        time_left = max(0, self.timer / self.FPS)
        timer_text = f"TIME: {time_left:.1f}"
        timer_color = (255, 100, 100) if time_left < 10 else self.COLOR_TEXT
        self._draw_text(timer_text, (self.WIDTH - 150, 10), self.font_ui, timer_color, self.COLOR_TEXT_SHADOW)
        
        if self.game_over:
            won = self.current_platform_index == len(self.platforms) - 1
            msg = "GOAL!" if won else "GAME OVER"
            color = self.COLOR_GOAL if won else (200, 50, 50)
            self._draw_text(msg, self.screen.get_rect().center, self.font_big, color, self.COLOR_TEXT_SHADOW, center=True)

    def _draw_text(self, text, pos, font, color, shadow_color, center=False):
        shadow_surf = font.render(text, True, shadow_color)
        text_surf = font.render(text, True, color)
        
        shadow_pos = (pos[0] + 2, pos[1] + 2)
        if center:
            shadow_pos = shadow_surf.get_rect(center=shadow_pos)
            pos = text_surf.get_rect(center=pos)

        self.screen.blit(shadow_surf, shadow_pos)
        self.screen.blit(text_surf, pos)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "time_left": max(0, self.timer / self.FPS),
        }

    def _create_particles(self, pos, count, color, p_type):
        for _ in range(count):
            life = self.np_random.integers(15, 30)
            if p_type == 'jump':
                vel = pygame.Vector2(self.np_random.uniform(-1, 1), self.np_random.uniform(0, 2))
            elif p_type == 'land':
                vel = pygame.Vector2(self.np_random.uniform(-2, 2), self.np_random.uniform(-1.5, -0.5))
            else: # generic
                vel = pygame.Vector2(self.np_random.uniform(-1, 1), self.np_random.uniform(-1, 1))

            self.particles.append({
                'pos': pygame.Vector2(pos),
                'vel': vel,
                'life': life,
                'max_life': life,
                'color': pygame.Color(*color)
            })

    def close(self):
        pygame.quit()

if __name__ == "__main__":
    # This block allows you to play the game directly
    # Set `auto_advance` to False to control frames with the 'n' key for debugging
    GameEnv.auto_advance = True

    # Unset the dummy video driver to allow for display
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]

    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((GameEnv.WIDTH, GameEnv.HEIGHT))
    pygame.display.set_caption("Platform Hopper")
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    # --- Main Game Loop ---
    while running:
        movement, space, shift = 0, 0, 0
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        if keys[pygame.K_SPACE]: space = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift = 1
        
        action = [movement, space, shift]
        
        manual_step = False
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if not GameEnv.auto_advance and event.type == pygame.KEYDOWN and event.key == pygame.K_n:
                 manual_step = True

        if GameEnv.auto_advance or manual_step:
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward

            if terminated:
                print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
                obs, info = env.reset()
                total_reward = 0

        # --- Rendering ---
        # The environment returns the rendered frame, so we just blit it
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        if GameEnv.auto_advance:
            clock.tick(GameEnv.FPS)

    env.close()