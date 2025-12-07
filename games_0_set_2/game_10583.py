import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T14:43:33.178373
# Source Brief: brief_00583.md
# Brief Index: 583
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    Cyberpunk brawlers trade word-based attacks in gravity-flipping, time-bending lexical combat.
    This Gymnasium environment features a 2D fighting game where the agent controls a character
    that can move horizontally, fire 'word' projectiles, and activate a time-slowing effect.
    The goal is to defeat the opponent by reducing their health to zero.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Engage in a fast-paced cyberpunk duel, attacking your opponent with 'word' projectiles "
        "while dodging their lexical assaults."
    )
    user_guide = (
        "Controls: ←→ to move. Press space to fire a 'word' projectile. "
        "Press shift to activate the time-slowing effect."
    )
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 30 # Assumed for smooth interpolation

        # Colors
        self.COLOR_BG = (10, 5, 25)
        self.COLOR_GRID = (30, 20, 50)
        self.COLOR_PLAYER = (0, 150, 255)
        self.COLOR_PLAYER_GLOW = (0, 150, 255, 50)
        self.COLOR_OPPONENT = (255, 50, 50)
        self.COLOR_OPPONENT_GLOW = (255, 50, 50, 50)
        self.COLOR_TIME_SLOW = (255, 220, 0)
        self.COLOR_UI_TEXT = (220, 220, 240)
        self.COLOR_UI_BG = (40, 40, 60, 180)
        
        # Player settings
        self.PLAYER_MAX_HEALTH = 100
        self.PLAYER_SPEED = 12
        self.PLAYER_FRICTION = 0.85
        self.PLAYER_ATTACK_COOLDOWN = 15 # steps
        self.PLAYER_TIME_SLOW_DURATION = 90 # steps (3 seconds)
        self.PLAYER_TIME_SLOW_COOLDOWN = 300 # steps (10 seconds)
        
        # Opponent settings
        self.OPPONENT_MAX_HEALTH = 100
        self.OPPONENT_ATTACK_INTERVAL = 90 # steps (3 seconds)

        # Word list for attacks
        self.WORD_LIST = ["GLI.TCH", "C0RR.UPT", "D.ECRYPT", "N.EURAL", "SYN.TH", "H.EX", "C.YBER", "V.OID"]
        
        # --- Gymnasium Spaces ---
        self.observation_space = Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("Consolas", 16, bold=True)
        self.font_word = pygame.font.SysFont("Consolas", 22, bold=True)
        self.font_game_over = pygame.font.SysFont("Consolas", 48, bold=True)

        # --- State Variables ---
        # These are initialized in reset()
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.player_state = {}
        self.opponent_state = {}
        self.player_projectiles = []
        self.opponent_projectiles = []
        self.particles = []
        self.prev_space_held = False
        self.prev_shift_held = False
        self.reward_this_step = 0
        
        # --- Initial Reset ---
        # self.reset() is called by the environment wrapper
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.reward_this_step = 0
        
        self.player_state = {
            "pos": pygame.Vector2(self.WIDTH / 4, self.HEIGHT - 50),
            "vel": pygame.Vector2(0, 0),
            "health": self.PLAYER_MAX_HEALTH,
            "attack_cooldown": 0,
            "time_slow_timer": 0,
            "time_slow_cooldown": 0,
            "word_idx": self.np_random.integers(0, len(self.WORD_LIST)),
        }
        
        self.opponent_state = {
            "pos": pygame.Vector2(self.WIDTH * 3 / 4, self.HEIGHT - 50),
            "health": self.OPPONENT_MAX_HEALTH,
            "attack_timer": self.np_random.integers(30, self.OPPONENT_ATTACK_INTERVAL),
            "word_idx": self.np_random.integers(0, len(self.WORD_LIST)),
        }
        
        self.player_projectiles = []
        self.opponent_projectiles = []
        self.particles = []
        
        self.prev_space_held = False
        self.prev_shift_held = False

        return self._get_observation(), self._get_info()

    def step(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        self.reward_this_step = 0
        
        if not self.game_over:
            # --- Update Game Logic ---
            self._handle_input(movement, space_held, shift_held)
            self._update_player()
            self._update_opponent()
            self._update_projectiles()
            self._update_particles()
            self._handle_collisions()
            
            self.steps += 1
        
        # --- Calculate Rewards and Termination ---
        reward = self.reward_this_step
        terminated = self._check_termination()
        truncated = self.steps >= 1800 # Truncate after 1 minute

        if terminated and not self.game_over:
            self.game_over = True
            if self.player_state["health"] <= 0:
                reward -= 100 # Loss penalty
            elif self.opponent_state["health"] <= 0:
                reward += 100 # Win bonus
        
        self.score += reward
        
        # Update previous button states for edge detection
        self.prev_space_held = space_held
        self.prev_shift_held = shift_held
        
        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _handle_input(self, movement, space_held, shift_held):
        # Movement
        if movement == 3: # Left
            self.player_state["vel"].x -= 2
        elif movement == 4: # Right
            self.player_state["vel"].x += 2

        # Attack (Spacebar press)
        if space_held and not self.prev_space_held and self.player_state["attack_cooldown"] <= 0:
            self._fire_projectile(is_player=True)
            self.player_state["attack_cooldown"] = self.PLAYER_ATTACK_COOLDOWN
            # sfx: player_shoot

        # Time Slow (Shift press)
        if shift_held and not self.prev_shift_held and self.player_state["time_slow_cooldown"] <= 0:
            self.player_state["time_slow_timer"] = self.PLAYER_TIME_SLOW_DURATION
            self.player_state["time_slow_cooldown"] = self.PLAYER_TIME_SLOW_COOLDOWN
            # sfx: time_slow_activate

    def _update_player(self):
        # Update cooldowns
        if self.player_state["attack_cooldown"] > 0:
            self.player_state["attack_cooldown"] -= 1
        if self.player_state["time_slow_timer"] > 0:
            self.player_state["time_slow_timer"] -= 1
        if self.player_state["time_slow_cooldown"] > 0:
            self.player_state["time_slow_cooldown"] -= 1
        
        # Update position and velocity
        self.player_state["pos"] += self.player_state["vel"]
        self.player_state["vel"] *= self.PLAYER_FRICTION
        self.player_state["vel"].x = max(-self.PLAYER_SPEED, min(self.PLAYER_SPEED, self.player_state["vel"].x))
        
        # Boundary checks
        self.player_state["pos"].x = max(25, min(self.WIDTH - 25, self.player_state["pos"].x))

    def _update_opponent(self):
        self.opponent_state["attack_timer"] -= 1
        if self.opponent_state["attack_timer"] <= 0:
            self._fire_projectile(is_player=False)
            self.opponent_state["attack_timer"] = self.OPPONENT_ATTACK_INTERVAL
            # sfx: opponent_shoot

    def _fire_projectile(self, is_player):
        if is_player:
            start_pos = self.player_state["pos"].copy()
            start_pos.y -= 30
            velocity = pygame.Vector2(0, -10)
            word = self.WORD_LIST[self.player_state["word_idx"]]
            self.player_projectiles.append({"pos": start_pos, "vel": velocity, "word": word})
            self.player_state["word_idx"] = (self.player_state["word_idx"] + 1) % len(self.WORD_LIST)
        else:
            start_pos = self.opponent_state["pos"].copy()
            start_pos.y -= 30
            velocity = pygame.Vector2(0, 10)
            word = self.WORD_LIST[self.opponent_state["word_idx"]]
            self.opponent_projectiles.append({"pos": start_pos, "vel": velocity, "word": word})
            self.opponent_state["word_idx"] = (self.opponent_state["word_idx"] + 1) % len(self.WORD_LIST)

    def _update_projectiles(self):
        time_scale = 0.3 if self.player_state["time_slow_timer"] > 0 else 1.0
        
        for proj in self.player_projectiles[:]:
            proj["pos"] += proj["vel"] * time_scale
            if proj["pos"].y < 0:
                self.player_projectiles.remove(proj)
        
        for proj in self.opponent_projectiles[:]:
            proj["pos"] += proj["vel"] * time_scale
            if proj["pos"].y > self.HEIGHT:
                self.opponent_projectiles.remove(proj)

    def _update_particles(self):
        for p in self.particles[:]:
            p["pos"] += p["vel"]
            p["life"] -= 1
            if p["life"] <= 0:
                self.particles.remove(p)

    def _handle_collisions(self):
        player_rect = pygame.Rect(self.player_state["pos"].x - 15, self.player_state["pos"].y - 15, 30, 30)
        opponent_rect = pygame.Rect(self.opponent_state["pos"].x - 15, self.opponent_state["pos"].y - 15, 30, 30)

        for proj in self.player_projectiles[:]:
            proj_rect = pygame.Rect(proj["pos"].x - 30, proj["pos"].y - 10, 60, 20)
            if proj_rect.colliderect(opponent_rect):
                damage = len(proj["word"])
                self.opponent_state["health"] -= damage
                self._create_impact_particles(proj["pos"], self.COLOR_OPPONENT)
                self.player_projectiles.remove(proj)
                
                is_time_slow_hit = self.player_state["time_slow_timer"] > 0
                self.reward_this_step += 1.0 if is_time_slow_hit else 0.1
                # sfx: opponent_hit
                
        for proj in self.opponent_projectiles[:]:
            proj_rect = pygame.Rect(proj["pos"].x - 30, proj["pos"].y - 10, 60, 20)
            if proj_rect.colliderect(player_rect):
                damage = len(proj["word"])
                self.player_state["health"] -= damage
                self._create_impact_particles(proj["pos"], self.COLOR_PLAYER)
                self.opponent_projectiles.remove(proj)
                self.reward_this_step -= 0.1
                # sfx: player_hit

    def _create_impact_particles(self, pos, color):
        for _ in range(20):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = pygame.Vector2(math.cos(angle) * speed, math.sin(angle) * speed)
            self.particles.append({
                "pos": pos.copy(),
                "vel": vel,
                "life": self.np_random.integers(10, 25),
                "color": color,
                "size": self.np_random.uniform(1, 3)
            })

    def _check_termination(self):
        return (
            self.player_state["health"] <= 0 or
            self.opponent_state["health"] <= 0
        )

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_background()
        
        if self.player_state["time_slow_timer"] > 0:
            self._render_time_slow_effect()
            
        self._render_game_elements()
        self._render_ui()
        
        if self.game_over:
            self._render_game_over()
            
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_background(self):
        for i in range(0, self.WIDTH, 40):
            pygame.draw.line(self.screen, self.COLOR_GRID, (i, 0), (i, self.HEIGHT), 1)
        for i in range(0, self.HEIGHT, 40):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, i), (self.WIDTH, i), 1)
        # Floor line
        pygame.draw.line(self.screen, self.COLOR_PLAYER, (0, self.HEIGHT - 20), (self.WIDTH, self.HEIGHT - 20), 2)

    def _render_time_slow_effect(self):
        # Yellow overlay
        overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
        alpha = int(90 * (self.player_state["time_slow_timer"] / self.PLAYER_TIME_SLOW_DURATION))
        overlay.fill((*self.COLOR_TIME_SLOW, alpha))
        self.screen.blit(overlay, (0, 0))
        
        # Screen wave effect
        pixels = pygame.surfarray.pixels3d(self.screen)
        amplitude = 3
        frequency = 0.05
        offset = self.steps * 0.5
        for y in range(self.HEIGHT):
            x_offset = int(amplitude * math.sin(y * frequency + offset))
            pixels[y,:] = np.roll(pixels[y,:], x_offset, axis=0)
        del pixels


    def _render_game_elements(self):
        # Render projectiles
        for proj in self.player_projectiles:
            self._render_word(proj["word"], proj["pos"], self.COLOR_PLAYER, self.COLOR_PLAYER_GLOW)
        for proj in self.opponent_projectiles:
            self._render_word(proj["word"], proj["pos"], self.COLOR_OPPONENT, self.COLOR_OPPONENT_GLOW)
            
        # Render particles
        for p in self.particles:
            alpha = int(255 * (p["life"] / 25))
            color = (*p["color"], alpha)
            pygame.draw.circle(self.screen, color, (int(p["pos"].x), int(p["pos"].y)), int(p["size"]))

        # Render characters
        self._render_character(self.player_state, self.COLOR_PLAYER, self.COLOR_PLAYER_GLOW, is_player=True)
        self._render_character(self.opponent_state, self.COLOR_OPPONENT, self.COLOR_OPPONENT_GLOW, is_player=False)

    def _render_character(self, state, color, glow_color, is_player):
        pos = state["pos"]
        # Glow
        size = 22
        points_glow = [
            (pos.x, pos.y - size),
            (pos.x - size, pos.y + size),
            (pos.x + size, pos.y + size),
        ]
        pygame.gfxdraw.aapolygon(self.screen, [(int(p[0]), int(p[1])) for p in points_glow], glow_color)
        pygame.gfxdraw.filled_polygon(self.screen, [(int(p[0]), int(p[1])) for p in points_glow], glow_color)
        
        # Main shape
        size = 15
        points_main = [
            (pos.x, pos.y - size),
            (pos.x - size, pos.y + size),
            (pos.x + size, pos.y + size),
        ]
        pygame.gfxdraw.aapolygon(self.screen, [(int(p[0]), int(p[1])) for p in points_main], color)
        pygame.gfxdraw.filled_polygon(self.screen, [(int(p[0]), int(p[1])) for p in points_main], color)

    def _render_word(self, text, pos, color, glow_color):
        # Glow effect by rendering multiple times
        for dx, dy in [(-1, -1), (1, -1), (-1, 1), (1, 1), (-2, 0), (2, 0), (0, -2), (0, 2)]:
            text_surf = self.font_word.render(text, True, glow_color)
            text_rect = text_surf.get_rect(center=(int(pos.x + dx), int(pos.y + dy)))
            self.screen.blit(text_surf, text_rect)
        
        # Main text
        text_surf = self.font_word.render(text, True, color)
        text_rect = text_surf.get_rect(center=(int(pos.x), int(pos.y)))
        self.screen.blit(text_surf, text_rect)

    def _render_ui(self):
        # Player UI
        self._render_bar(10, 10, 200, 20, self.player_state["health"], self.PLAYER_MAX_HEALTH, self.COLOR_PLAYER, f"PLAYER HP")
        self._render_bar(10, 35, 200, 10, self.PLAYER_TIME_SLOW_DURATION - self.player_state["time_slow_timer"], self.PLAYER_TIME_SLOW_DURATION, self.COLOR_TIME_SLOW, "TIME", show_percent=False)
        self._render_bar(10, 50, 200, 10, self.PLAYER_TIME_SLOW_COOLDOWN - self.player_state["time_slow_cooldown"], self.PLAYER_TIME_SLOW_COOLDOWN, (100,100,120), "READY", show_percent=False)
        
        # Opponent UI
        self._render_bar(self.WIDTH - 210, 10, 200, 20, self.opponent_state["health"], self.OPPONENT_MAX_HEALTH, self.COLOR_OPPONENT, f"OPPONENT HP")
        
        # Word display
        player_word = self.WORD_LIST[self.player_state["word_idx"]]
        word_surf = self.font_ui.render(f"NEXT: {player_word}", True, self.COLOR_UI_TEXT)
        self.screen.blit(word_surf, (10, 65))
        
        opponent_word = self.WORD_LIST[self.opponent_state["word_idx"]]
        word_surf = self.font_ui.render(f"NEXT: {opponent_word}", True, self.COLOR_UI_TEXT)
        text_rect = word_surf.get_rect(topright=(self.WIDTH - 10, 35))
        self.screen.blit(word_surf, text_rect)

    def _render_bar(self, x, y, w, h, value, max_value, color, label, show_percent=True):
        value = max(0, value)
        # Background
        bg_rect = pygame.Rect(x, y, w, h)
        pygame.draw.rect(self.screen, self.COLOR_UI_BG, bg_rect, border_radius=3)
        # Fill
        fill_w = (value / max_value) * w if max_value > 0 else 0
        fill_rect = pygame.Rect(x, y, int(fill_w), h)
        pygame.draw.rect(self.screen, color, fill_rect, border_radius=3)
        # Label
        if show_percent:
            text = f"{label}: {int(value)}/{max_value}"
        else:
            text = label
        text_surf = self.font_ui.render(text, True, self.COLOR_UI_TEXT)
        text_rect = text_surf.get_rect(center=bg_rect.center)
        self.screen.blit(text_surf, text_rect)

    def _render_game_over(self):
        overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 180))
        self.screen.blit(overlay, (0, 0))
        
        if self.player_state["health"] <= 0:
            text = "SYSTEM FAILURE"
            color = self.COLOR_OPPONENT
        elif self.opponent_state["health"] <= 0:
            text = "TARGET NEUTRALIZED"
            color = self.COLOR_PLAYER
        else:
            text = "COMBAT INCONCLUSIVE"
            color = self.COLOR_UI_TEXT
        
        text_surf = self.font_game_over.render(text, True, color)
        text_rect = text_surf.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
        self.screen.blit(text_surf, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "player_health": self.player_state["health"],
            "opponent_health": self.opponent_state["health"],
        }
        
    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # This block allows you to play the game manually for testing
    # To use, you'll need to `pip install pygame`
    # And unset the dummy video driver
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]
        
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Cyberpunk Lexical Combat")
    clock = pygame.time.Clock()
    
    terminated = False
    
    # --- Main Game Loop for Manual Play ---
    while not terminated:
        movement, space_held, shift_held = 0, 0, 0
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT] or keys[pygame.K_a]:
            movement = 3
        elif keys[pygame.K_RIGHT] or keys[pygame.K_d]:
            movement = 4
        
        if keys[pygame.K_SPACE]:
            space_held = 1
        
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            shift_held = 1
            
        action = [movement, space_held, shift_held]
        
        obs, reward, term, trunc, info = env.step(action)
        terminated = term or trunc
        
        # Render the observation to the display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(env.FPS)
        
    env.close()