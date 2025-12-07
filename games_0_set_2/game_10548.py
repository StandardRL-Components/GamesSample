import gymnasium as gym
import os
import pygame
import pygame.gfxdraw
import math
import numpy as np
from gymnasium.spaces import MultiDiscrete
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}
    
    game_description = (
        "Navigate a shifting dreamscape, collect soul fragments to power your abilities, and find the path to the next realm. "
        "Phase through obstacles or reveal hidden paths to survive."
    )
    user_guide = (
        "Use arrow keys to move. Press space to spend fragments and reveal hidden paths. "
        "Press shift to spend fragments and phase through obstacles."
    )
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 30
        self.FINAL_REALM = 5
        self.MAX_STEPS = 2500

        # Colors
        self.COLOR_BG = (10, 5, 25)
        self.COLOR_PLAYER = (255, 255, 255)
        self.COLOR_PLAYER_PHASE = (180, 220, 255)
        self.COLOR_OBSTACLE = (255, 120, 0)
        self.COLOR_FRAGMENT = (200, 100, 255)
        self.COLOR_PATH_REVEALED = (0, 150, 255)
        self.COLOR_GOAL = (100, 255, 100)
        self.COLOR_UI_TEXT = (220, 220, 220)
        self.COLOR_UI_ICON_REVEAL = (0, 150, 255)
        self.COLOR_UI_ICON_PHASE = (255, 120, 0)

        # Player settings
        self.PLAYER_SPEED = 8
        self.PLAYER_SIZE = 10
        self.PLAYER_LERP_FACTOR = 0.25

        # Ability settings
        self.REVEAL_COST = 5
        self.REVEAL_DURATION = 5 * self.FPS  # 5 seconds
        self.PHASE_COST = 10
        self.PHASE_DURATION = 2 * self.FPS  # 2 seconds
        self.MAX_SOUL_FRAGMENTS = 100

        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("monospace", 18, bold=True)
        self.font_realm = pygame.font.SysFont("monospace", 36, bold=True)

        # --- State Variables ---
        self.steps = None
        self.score = None
        self.game_over = None
        self.player_pos = None
        self.player_target_pos = None
        self.soul_fragments = None
        self.current_realm = None
        self.reveal_timer = None
        self.phase_timer = None
        self.space_pressed_last_frame = None
        self.shift_pressed_last_frame = None
        self.obstacles = []
        self.fragments = []
        self.hidden_paths = []
        self.particles = []
        self.goal = None
        self.phased_this_step = set()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.current_realm = 1
        self.soul_fragments = 20
        self.reveal_timer = 0
        self.phase_timer = 0
        self.space_pressed_last_frame = False
        self.shift_pressed_last_frame = False
        
        self.particles.clear()
        
        start_pos = [self.WIDTH * 0.15, self.HEIGHT / 2]
        self.player_pos = list(start_pos)
        self.player_target_pos = list(start_pos)

        self._generate_level()

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        reward = 0.0

        # --- Handle Input and Abilities ---
        reward += self._handle_input(action)

        # --- Update Game State ---
        self._update_game_state()
        
        # --- Handle Collisions ---
        reward += self._handle_collisions()
        
        self.score += reward

        # --- Check Termination ---
        terminated = self._check_termination()
        truncated = self.steps >= self.MAX_STEPS
        if truncated:
            self.game_over = True

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _generate_level(self):
        self.obstacles.clear()
        self.fragments.clear()
        self.hidden_paths.clear()
        
        main_path = pygame.Rect(self.WIDTH * 0.1, self.HEIGHT / 2 - 25, self.WIDTH * 0.8, 50)
        
        path_y = self.HEIGHT * 0.2 if self.np_random.random() > 0.5 else self.HEIGHT * 0.8
        hidden_path = pygame.Rect(self.WIDTH * 0.3, path_y - 25, self.WIDTH * 0.4, 50)
        self.hidden_paths.append(hidden_path)

        self.goal = pygame.Rect(self.WIDTH - 60, self.HEIGHT / 2 - 20, 20, 40)
        
        for _ in range(10):
            self.fragments.append(pygame.Rect(
                self.np_random.integers(main_path.left, main_path.right),
                self.np_random.integers(main_path.top, main_path.bottom), 8, 8))
        for _ in range(5):
            self.fragments.append(pygame.Rect(
                self.np_random.integers(hidden_path.left, hidden_path.right),
                self.np_random.integers(hidden_path.top, hidden_path.bottom), 8, 8))

        obstacle_count = 10 + int(self.current_realm * 5)
        for _ in range(obstacle_count):
            while True:
                obs_rect = pygame.Rect(
                    self.np_random.integers(0, self.WIDTH - 20),
                    self.np_random.integers(0, self.HEIGHT - 20),
                    self.np_random.integers(15, 30), self.np_random.integers(15, 30))
                if not obs_rect.colliderect(main_path) and \
                   not obs_rect.colliderect(hidden_path) and \
                   not obs_rect.colliderect(self.goal):
                    self.obstacles.append(obs_rect)
                    break

    def _handle_input(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        reward = 0.0

        if movement == 1: self.player_target_pos[1] -= self.PLAYER_SPEED
        elif movement == 2: self.player_target_pos[1] += self.PLAYER_SPEED
        elif movement == 3: self.player_target_pos[0] -= self.PLAYER_SPEED
        elif movement == 4: self.player_target_pos[0] += self.PLAYER_SPEED

        self.player_target_pos[0] = np.clip(self.player_target_pos[0], 0, self.WIDTH)
        self.player_target_pos[1] = np.clip(self.player_target_pos[1], 0, self.HEIGHT)

        if space_held and not self.space_pressed_last_frame:
            if self.soul_fragments >= self.REVEAL_COST and self.reveal_timer == 0:
                self.soul_fragments -= self.REVEAL_COST
                self.reveal_timer = self.REVEAL_DURATION
                reward += 1.0
                self._create_effect(self.player_pos, self.COLOR_PATH_REVEALED, 20)

        if shift_held and not self.shift_pressed_last_frame:
            if self.soul_fragments >= self.PHASE_COST and self.phase_timer == 0:
                self.soul_fragments -= self.PHASE_COST
                self.phase_timer = self.PHASE_DURATION
                self._create_effect(self.player_pos, self.COLOR_PLAYER_PHASE, 20)

        self.space_pressed_last_frame = space_held
        self.shift_pressed_last_frame = shift_held
        return reward

    def _update_game_state(self):
        self.player_pos[0] += (self.player_target_pos[0] - self.player_pos[0]) * self.PLAYER_LERP_FACTOR
        self.player_pos[1] += (self.player_target_pos[1] - self.player_pos[1]) * self.PLAYER_LERP_FACTOR

        if self.reveal_timer > 0: self.reveal_timer -= 1
        if self.phase_timer > 0: self.phase_timer -= 1

        if self.steps % 2 == 0:
            p_color = self.COLOR_PLAYER_PHASE if self.phase_timer > 0 else self.COLOR_PLAYER
            self.particles.append({
                "pos": list(self.player_pos),
                "vel": [0, 0],
                "size": self.PLAYER_SIZE * 0.8,
                "lifetime": 15,
                "color": p_color,
                "type": "trail"
            })

        for p in self.particles[:]:
            p["pos"][0] += p["vel"][0]
            p["pos"][1] += p["vel"][1]
            p["lifetime"] -= 1
            if p["type"] == "trail": p["size"] *= 0.95
            if p["lifetime"] <= 0:
                self.particles.remove(p)
                
        self.phased_this_step.clear()

    def _handle_collisions(self):
        reward = 0.0
        player_rect = pygame.Rect(
            self.player_pos[0] - self.PLAYER_SIZE / 2,
            self.player_pos[1] - self.PLAYER_SIZE / 2,
            self.PLAYER_SIZE, self.PLAYER_SIZE
        )

        for frag in self.fragments[:]:
            if player_rect.colliderect(frag):
                self.fragments.remove(frag)
                self.soul_fragments = min(self.MAX_SOUL_FRAGMENTS, self.soul_fragments + 1)
                reward += 0.1
                self._create_effect(frag.center, self.COLOR_FRAGMENT, 10)

        for obst in self.obstacles:
            if player_rect.colliderect(obst):
                if self.phase_timer > 0:
                    if id(obst) not in self.phased_this_step:
                        reward += 2.0
                        self.phased_this_step.add(id(obst))
                else:
                    if self.soul_fragments > 0:
                        self.soul_fragments -= 1
                    reward -= 0.5
                    self._create_effect(player_rect.center, self.COLOR_OBSTACLE, 15)
                    
        if player_rect.colliderect(self.goal):
            self.current_realm += 1
            if self.current_realm > self.FINAL_REALM:
                reward += 100.0
                self.game_over = True
            else:
                reward += 25.0
                self._generate_level()
                start_pos = [self.WIDTH * 0.15, self.HEIGHT / 2]
                self.player_pos = list(start_pos)
                self.player_target_pos = list(start_pos)
        
        return reward

    def _check_termination(self):
        if self.game_over:
            return True
        if self.soul_fragments <= 0:
            self.score -= 100.0
            self.game_over = True
            return True
        return False

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2))

    def _render_game(self):
        for path in self.hidden_paths:
            if self.reveal_timer > 0:
                reveal_alpha = int(150 * (self.reveal_timer / self.REVEAL_DURATION))
                s = pygame.Surface(path.size, pygame.SRCALPHA)
                s.fill((*self.COLOR_PATH_REVEALED, reveal_alpha // 4))
                self.screen.blit(s, path.topleft)
                pygame.draw.rect(self.screen, (*self.COLOR_PATH_REVEALED, reveal_alpha), path, 2)
            else:
                for _ in range(3):
                    pos = (
                        path.left + self.np_random.random() * path.width,
                        path.top + self.np_random.random() * path.height
                    )
                    pygame.gfxdraw.pixel(self.screen, int(pos[0]), int(pos[1]), (255, 255, 255, 20))

        pygame.draw.rect(self.screen, self.COLOR_GOAL, self.goal, 2)
        
        for obst in self.obstacles:
            pulse = (math.sin(self.steps * 0.1 + id(obst)) + 1) / 2
            color = (*self.COLOR_OBSTACLE, int(150 + 105 * pulse))
            s = pygame.Surface(obst.size, pygame.SRCALPHA)
            pygame.draw.rect(s, color, s.get_rect())
            self.screen.blit(s, obst.topleft)

        for frag in self.fragments:
            self._draw_glowing_circle(frag.center, frag.width, self.COLOR_FRAGMENT)

        for p in self.particles:
            max_lifetime = 20.0
            alpha = int(255 * (p["lifetime"] / max_lifetime))
            color = (*p["color"], alpha)
            pos = (int(p["pos"][0]), int(p["pos"][1]))
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], int(p["size"]), color)

        p_color = self.COLOR_PLAYER_PHASE if self.phase_timer > 0 else self.COLOR_PLAYER
        self._draw_glowing_circle(self.player_pos, self.PLAYER_SIZE, p_color)
        if self.phase_timer > 0:
            phase_alpha = 100 * (self.phase_timer / self.PHASE_DURATION)
            pygame.gfxdraw.aacircle(self.screen, int(self.player_pos[0]), int(self.player_pos[1]),
                                    int(self.PLAYER_SIZE * 1.5), (*self.COLOR_PLAYER_PHASE, int(phase_alpha)))

    def _render_ui(self):
        text_surf = self.font_ui.render(f"Fragments: {self.soul_fragments}", True, self.COLOR_UI_TEXT)
        self.screen.blit(text_surf, (10, 10))
        
        self._draw_ability_icon(20, 40, self.REVEAL_COST, self.COLOR_UI_ICON_REVEAL, self.reveal_timer, self.REVEAL_DURATION)
        self._draw_ability_icon(60, 40, self.PHASE_COST, self.COLOR_UI_ICON_PHASE, self.phase_timer, self.PHASE_DURATION)

        if self.steps < 90:
            alpha = 255 if self.steps < 60 else 255 * (90 - self.steps) / 30
            realm_text = self.font_realm.render(f"Dream Realm {self.current_realm}", True, self.COLOR_UI_TEXT)
            realm_text.set_alpha(alpha)
            self.screen.blit(realm_text, realm_text.get_rect(center=(self.WIDTH/2, self.HEIGHT/2)))

    def _draw_glowing_circle(self, pos, radius, color):
        pos_int = (int(pos[0]), int(pos[1]))
        for i in range(int(radius), 0, -2):
            alpha = int(150 * (1 - (i / radius)))
            pygame.gfxdraw.filled_circle(self.screen, pos_int[0], pos_int[1], i, (*color, alpha))
        pygame.gfxdraw.filled_circle(self.screen, pos_int[0], pos_int[1], int(radius*0.5), color)
        pygame.gfxdraw.aacircle(self.screen, pos_int[0], pos_int[1], int(radius*0.5), color)

    def _create_effect(self, pos, color, num_particles):
        for _ in range(num_particles):
            angle = self.np_random.random() * 2 * math.pi
            speed = self.np_random.random() * 2 + 1
            self.particles.append({
                "pos": list(pos),
                "vel": [math.cos(angle) * speed, math.sin(angle) * speed],
                "size": self.np_random.random() * 3 + 1,
                "lifetime": 20,
                "color": color,
                "type": "burst"
            })
            
    def _draw_ability_icon(self, x, y, cost, color, timer, duration):
        icon_rect = pygame.Rect(x, y, 30, 30)
        
        if timer > 0:
            s = pygame.Surface((30, 30), pygame.SRCALPHA)
            s.fill((50, 50, 50, 180))
            self.screen.blit(s, (x, y))
            
            angle = -math.pi/2 + (2 * math.pi * (1 - timer / duration))
            pygame.draw.arc(self.screen, color, icon_rect.inflate(4, 4), -math.pi/2, angle, 2)
        else:
            cost_color = self.COLOR_UI_TEXT if self.soul_fragments >= cost else (255, 50, 50)
            cost_surf = self.font_ui.render(str(cost), True, cost_color)
            self.screen.blit(cost_surf, cost_surf.get_rect(center=icon_rect.center))

        pygame.draw.rect(self.screen, color, icon_rect, 1)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "realm": self.current_realm,
            "soul_fragments": self.soul_fragments
        }

    def close(self):
        pygame.quit()


if __name__ == "__main__":
    # This block will not be run by the evaluation environment.
    # It is for human interaction and debugging.
    
    # Un-set the headless driver for local play
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]

    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    pygame.display.set_caption("Dreamscape Drifter")
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    clock = pygame.time.Clock()
    
    total_reward = 0
    
    while not done:
        keys = pygame.key.get_pressed()
        movement = 0
        if keys[pygame.K_UP] or keys[pygame.K_w]: movement = 1
        elif keys[pygame.K_DOWN] or keys[pygame.K_s]: movement = 2
        elif keys[pygame.K_LEFT] or keys[pygame.K_a]: movement = 3
        elif keys[pygame.K_RIGHT] or keys[pygame.K_d]: movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]

        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward

        # The observation is already the rendered screen
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
        
        clock.tick(env.FPS)
        
        if done:
            print(f"Episode Finished. Final Score: {info['score']:.2f}, Steps: {info['steps']}, Realm: {info['realm']}")

    env.close()