
# Generated: 2025-08-28T03:12:58.564196
# Source Brief: brief_04854.md
# Brief Index: 4854

        
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


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: Press space to jump over the monsters."
    )

    game_description = (
        "Survive as long as possible by jumping over an endless stream of procedurally generated monsters in a vibrant, retro-styled world."
    )

    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    FPS = 60
    MAX_TIME_SECONDS = 90
    MAX_STEPS = MAX_TIME_SECONDS * FPS

    # --- Colors ---
    COLOR_BG_TOP = (40, 20, 80)
    COLOR_BG_BOTTOM = (80, 50, 120)
    COLOR_GROUND = (30, 60, 30)
    COLOR_PLAYER = (255, 220, 0)
    COLOR_PLAYER_EYE = (0, 0, 0)
    COLOR_PARTICLE = (255, 255, 255)
    COLOR_UI_TEXT = (255, 255, 255)
    COLOR_UI_SHADOW = (50, 50, 50)
    
    MONSTER_DEFINITIONS = {
        1: {'width': 40, 'height': 30, 'color': (0, 200, 80)},    # Green
        2: {'width': 35, 'height': 60, 'color': (220, 50, 50)},    # Red
        3: {'width': 50, 'height': 45, 'color': (80, 80, 220)},    # Blue
    }

    # --- Physics & Gameplay ---
    GROUND_Y = 350
    GRAVITY = 0.5
    JUMP_STRENGTH = -12
    PLAYER_X = 100
    PLAYER_BASE_WIDTH = 30
    PLAYER_BASE_HEIGHT = 50
    INITIAL_MONSTER_SPEED = 2.0
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        
        self.ui_font_large = pygame.font.SysFont("monospace", 48, bold=True)
        self.ui_font_small = pygame.font.SysFont("monospace", 24, bold=True)
        
        self.monsters = []
        self.particles = []
        self.np_random = None

        # These attributes are initialized in reset()
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False
        self.player_y = 0
        self.player_vy = 0
        self.player_on_ground = False
        self.player_squash = 0
        self.player_rect = pygame.Rect(0, 0, 0, 0)
        self.just_jumped = False
        self.monster_id_counter = 0
        self.monsters_passed = set()
        self.monster_speed = 0
        self.monster_spawn_timer = 0
        self.monster_types_available = []

        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.np_random = np.random.default_rng(seed)
        else:
            # It's good practice to have an RNG instance even if seed is None
            if self.np_random is None:
                self.np_random = np.random.default_rng()

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False

        self.player_y = self.GROUND_Y - self.PLAYER_BASE_HEIGHT
        self.player_vy = 0
        self.player_on_ground = True
        self.player_squash = 0
        self.player_rect = pygame.Rect(
            self.PLAYER_X, self.player_y, self.PLAYER_BASE_WIDTH, self.PLAYER_BASE_HEIGHT
        )
        self.just_jumped = False

        self.monsters.clear()
        self.particles.clear()
        self.monster_id_counter = 0
        self.monsters_passed = set()
        self.monster_speed = self.INITIAL_MONSTER_SPEED
        self.monster_spawn_timer = 60 # Start spawning monsters after 1 second
        self.monster_types_available = [1]
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        reward = 0.1  # Survival reward per frame
        terminated = False

        if self.game_over:
            # If the game is already over, do nothing but advance frame counter
            self.steps += 1
            return self._get_observation(), 0, True, False, self._get_info()

        # --- Handle Input ---
        if space_held and self.player_on_ground and not self.just_jumped:
            self.player_vy = self.JUMP_STRENGTH
            self.player_on_ground = False
            self.just_jumped = True
            # SFX: Jump

        if not space_held:
            self.just_jumped = False
            
        # --- Update Game State ---
        self._update_player()
        self._update_monsters()
        self._update_particles()
        self._update_difficulty()

        # --- Check for Pass Reward ---
        for monster in self.monsters:
            monster_center_x = monster['rect'].x + monster['rect'].width / 2
            if self.PLAYER_X > monster_center_x and monster['id'] not in self.monsters_passed:
                reward += 1
                self.monsters_passed.add(monster['id'])
                # SFX: Score point

        # --- Check Collisions ---
        for monster in self.monsters:
            if self.player_rect.colliderect(monster['rect']):
                self.game_over = True
                reward = -100
                # SFX: Player hit
                break
        
        self.steps += 1
        self.score += reward
        
        # --- Check Termination Conditions ---
        if self.game_over:
            terminated = True
        elif self.steps >= self.MAX_STEPS:
            self.win = True
            terminated = True
            reward += 100 # Bonus for winning
            self.score += 100
            # SFX: Win
        
        return self._get_observation(), reward, terminated, False, self._get_info()

    def _update_player(self):
        # Apply gravity
        if not self.player_on_ground:
            self.player_vy += self.GRAVITY
            self.player_y += self.player_vy

        # Check for landing
        if self.player_y >= self.GROUND_Y - self.PLAYER_BASE_HEIGHT:
            if not self.player_on_ground: # Just landed
                self.player_squash = 10 # frames
                self._create_landing_particles()
                # SFX: Land
            self.player_y = self.GROUND_Y - self.PLAYER_BASE_HEIGHT
            self.player_vy = 0
            self.player_on_ground = True
            
        if self.player_squash > 0:
            self.player_squash -= 1

        self.player_rect.y = int(self.player_y)

    def _update_monsters(self):
        # Spawn new monsters
        self.monster_spawn_timer -= 1
        if self.monster_spawn_timer <= 0:
            self._spawn_monster()
            spawn_interval = self.np_random.integers(45, 90)
            self.monster_spawn_timer = max(20, int(spawn_interval - self.monster_speed * 5))


        # Move and remove old monsters
        for monster in self.monsters[:]:
            monster['rect'].x -= self.monster_speed
            if monster['rect'].right < 0:
                self.monsters.remove(monster)

    def _spawn_monster(self):
        monster_type_id = self.np_random.choice(self.monster_types_available)
        m_def = self.MONSTER_DEFINITIONS[monster_type_id]
        
        monster_rect = pygame.Rect(
            self.SCREEN_WIDTH,
            self.GROUND_Y - m_def['height'],
            m_def['width'],
            m_def['height']
        )
        self.monsters.append({
            'rect': monster_rect,
            'color': m_def['color'],
            'id': self.monster_id_counter
        })
        self.monster_id_counter += 1

    def _update_particles(self):
        for p in self.particles[:]:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['vel'][1] += 0.1 # particle gravity
            p['life'] -= 1
            if p['life'] <= 0:
                self.particles.remove(p)

    def _update_difficulty(self):
        # Increase speed every 5 seconds (300 frames)
        if self.steps > 0 and self.steps % 300 == 0:
            self.monster_speed += 0.05
        
        # Introduce new monster types at 30s and 60s
        if self.steps == 1800: # 30 seconds
            if 2 not in self.monster_types_available: self.monster_types_available.append(2)
        if self.steps == 3600: # 60 seconds
            if 3 not in self.monster_types_available: self.monster_types_available.append(3)

    def _create_landing_particles(self):
        for _ in range(5):
            angle = self.np_random.uniform(math.pi * 1.1, math.pi * 1.9)
            speed = self.np_random.uniform(1, 3)
            particle = {
                'pos': [self.player_rect.centerx, self.player_rect.bottom],
                'vel': [math.cos(angle) * speed, math.sin(angle) * speed],
                'life': self.np_random.integers(15, 30),
                'radius': self.np_random.uniform(2, 4)
            }
            self.particles.append(particle)

    def _get_observation(self):
        self._render_background()
        self._render_ground()
        self._render_particles()
        self._render_monsters()
        self._render_player()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_background(self):
        for y in range(self.SCREEN_HEIGHT):
            interp = y / self.SCREEN_HEIGHT
            color = (
                self.COLOR_BG_TOP[0] * (1 - interp) + self.COLOR_BG_BOTTOM[0] * interp,
                self.COLOR_BG_TOP[1] * (1 - interp) + self.COLOR_BG_BOTTOM[1] * interp,
                self.COLOR_BG_TOP[2] * (1 - interp) + self.COLOR_BG_BOTTOM[2] * interp,
            )
            pygame.draw.line(self.screen, color, (0, y), (self.SCREEN_WIDTH, y))

    def _render_ground(self):
        pygame.draw.rect(self.screen, self.COLOR_GROUND, (0, self.GROUND_Y, self.SCREEN_WIDTH, self.SCREEN_HEIGHT - self.GROUND_Y))

    def _render_particles(self):
        for p in self.particles:
            radius = int(p['radius'] * (p['life'] / 30.0))
            if radius > 0:
                pos = (int(p['pos'][0]), int(p['pos'][1]))
                pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], radius, self.COLOR_PARTICLE)
                pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], radius, self.COLOR_PARTICLE)

    def _render_monsters(self):
        for monster in self.monsters:
            pygame.draw.rect(self.screen, monster['color'], monster['rect'])

    def _render_player(self):
        w, h = self.PLAYER_BASE_WIDTH, self.PLAYER_BASE_HEIGHT
        y_pos = self.player_y
        
        # Squash and stretch animation
        if self.player_squash > 0:
            squash_factor = 1.0 - 0.5 * math.sin((self.player_squash / 10.0) * math.pi)
            w *= (2.0 - squash_factor)
            h *= squash_factor
            y_pos += self.PLAYER_BASE_HEIGHT - h
        elif not self.player_on_ground:
            stretch = min(1.3, 1.0 - self.player_vy * 0.02)
            w *= (2.0 - stretch)
            h *= stretch
            y_pos += (self.PLAYER_BASE_HEIGHT - h) / 2

        visual_rect = pygame.Rect(self.PLAYER_X, y_pos, w, h)
        
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, visual_rect, border_radius=4)
        
        # Eye for direction
        eye_x = visual_rect.centerx + 5
        eye_y = visual_rect.centery - 10
        pygame.draw.circle(self.screen, self.COLOR_PLAYER_EYE, (int(eye_x), int(eye_y)), 3)

    def _render_ui(self):
        # Score
        score_text = f"SCORE: {int(self.score)}"
        self._draw_text(score_text, (20, 10), self.ui_font_small)

        # Timer
        time_left = max(0, (self.MAX_STEPS - self.steps) / self.FPS)
        timer_text = f"TIME: {time_left:.1f}"
        self._draw_text(timer_text, (self.SCREEN_WIDTH - 150, 10), self.ui_font_small)

        # Game Over / Win message
        if self.game_over:
            self._draw_text("GAME OVER", (self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT // 2 - 50), self.ui_font_large, center=True)
        elif self.win:
            self._draw_text("YOU WIN!", (self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT // 2 - 50), self.ui_font_large, center=True)

    def _draw_text(self, text, pos, font, color=COLOR_UI_TEXT, shadow_color=COLOR_UI_SHADOW, center=False):
        text_surf = font.render(text, True, color)
        text_rect = text_surf.get_rect()
        shadow_surf = font.render(text, True, shadow_color)
        shadow_rect = shadow_surf.get_rect()
        
        if center:
            text_rect.center = pos
            shadow_rect.center = (pos[0] + 2, pos[1] + 2)
        else:
            text_rect.topleft = pos
            shadow_rect.topleft = (pos[0] + 2, pos[1] + 2)
            
        self.screen.blit(shadow_surf, shadow_rect)
        self.screen.blit(text_surf, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "time_left": max(0, (self.MAX_STEPS - self.steps) / self.FPS),
            "win": self.win,
        }

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (400, 640, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (400, 640, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (400, 640, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game with keyboard controls
    # Not required by the spec, but useful for testing
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    pygame.display.set_caption("Jumper Game")
    clock = pygame.time.Clock()
    
    terminated = False
    total_reward = 0
    
    while not terminated:
        # --- Human Controls ---
        keys = pygame.key.get_pressed()
        space_held = keys[pygame.K_SPACE]
        
        action = [0, 1 if space_held else 0, 0] # Movement=None, Space=pressed/released, Shift=released
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                total_reward = 0

        # --- Environment Step ---
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # --- Rendering ---
        # The observation is already a rendered frame, so we just need to display it
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(env.FPS)
        
    print(f"Game Over! Final Score: {info['score']:.2f}, Steps: {info['steps']}")
    env.close()