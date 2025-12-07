import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T12:21:30.474259
# Source Brief: brief_01641.md
# Brief Index: 1641
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
        "Survive a void of hostile specters by absorbing them to match their color. "
        "Manage your essence, unlock powerful abilities, and enter portals for bonuses to achieve a high score."
    )
    user_guide = (
        "Controls: Use ↑↓←→ arrow keys to move. Press space to absorb specters or activate an ability. "
        "Press shift to cycle between unlocked abilities."
    )
    auto_advance = True

    # --- CONSTANTS ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    MAX_STEPS = 1000

    # Colors
    COLOR_BG_TOP = (10, 5, 25)
    COLOR_BG_BOTTOM = (30, 10, 50)
    COLOR_PLAYER_NEUTRAL = (255, 255, 255)
    COLOR_SPECTER_R = (255, 50, 50)
    COLOR_SPECTER_G = (50, 255, 50)
    COLOR_SPECTER_B = (80, 80, 255)
    COLOR_PORTAL_C = (50, 255, 255)
    COLOR_PORTAL_M = (255, 50, 255)
    COLOR_PORTAL_Y = (255, 255, 50)
    COLOR_UI_TEXT = (220, 220, 240)
    COLOR_UI_BAR = (50, 200, 255)
    COLOR_UI_BAR_BG = (40, 40, 80)
    COLOR_SHIELD = (100, 200, 255, 100) # RGBA

    SPECTER_COLORS = [COLOR_SPECTER_R, COLOR_SPECTER_G, COLOR_SPECTER_B]
    PORTAL_COLORS = [COLOR_PORTAL_C, COLOR_PORTAL_M, COLOR_PORTAL_Y]

    # Player
    PLAYER_SPEED = 4.0
    PLAYER_RADIUS = 12
    PLAYER_GLOW_RADIUS = 30

    # Specters
    SPECTER_RADIUS = 10
    ABSORB_RADIUS = 50
    INITIAL_SPECTER_SPAWN_RATE = 50 # steps
    INITIAL_SPECTER_SPEED = 1.0
    MAX_SPECTERS = 20

    # Essence
    MAX_ESSENCE = 200
    INITIAL_ESSENCE = 100
    ESSENCE_DECAY_RATE = 0.05
    ESSENCE_FROM_ABSORB = 25
    ESSENCE_PENALTY_MISMATCH = -50

    # Portals
    PORTAL_RADIUS = 25
    PORTAL_SPAWN_TIME = 300 # steps
    PORTAL_LIFESPAN = 400 # steps
    PORTAL_ESSENCE_BONUS = 50

    # Abilities
    ABILITY_UNLOCK_SHIELD = 50
    ABILITY_UNLOCK_DASH = 100
    ABILITY_SHIELD_COST = 30
    ABILITY_SHIELD_DURATION = 90 # steps
    ABILITY_SHIELD_COOLDOWN = 200
    ABILITY_DASH_COST = 15
    ABILITY_DASH_SPEED_MULTIPLIER = 4.0
    ABILITY_DASH_DURATION = 8 # steps
    ABILITY_DASH_COOLDOWN = 150

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
        self.font_small = pygame.font.SysFont("Consolas", 16, bold=True)
        self.font_large = pygame.font.SysFont("Consolas", 24, bold=True)
        
        self.render_mode = render_mode
        self.player_pos = np.array([0.0, 0.0])
        self.player_color_index = -1
        self.player_velocity = np.array([0.0, 0.0])
        self.essence = 0.0
        self.score = 0.0
        self.steps = 0
        self.game_over = False
        
        self.specters = []
        self.portals = []
        self.particles = []
        
        self.specter_spawn_timer = 0
        self.specter_spawn_rate = self.INITIAL_SPECTER_SPAWN_RATE
        self.specter_speed = self.INITIAL_SPECTER_SPEED
        self.portal_spawn_timer = self.PORTAL_SPAWN_TIME
        
        self.last_space_held = False
        self.last_shift_held = False
        
        self.abilities = {}
        self.selected_ability = None

        # This call will initialize all state variables
        # self.reset() # reset is called by the wrapper/runner
        
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.player_pos = np.array([self.SCREEN_WIDTH / 2.0, self.SCREEN_HEIGHT / 2.0])
        self.player_color_index = -1 # -1 for neutral/white
        self.player_velocity = np.array([0.0, 0.0])
        self.essence = self.INITIAL_ESSENCE
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        
        self.specters = []
        self.portals = []
        self.particles = []
        
        self.specter_spawn_timer = 0
        self.specter_spawn_rate = self.INITIAL_SPECTER_SPAWN_RATE
        self.specter_speed = self.INITIAL_SPECTER_SPEED
        self.portal_spawn_timer = self.PORTAL_SPAWN_TIME
        
        self.last_space_held = False
        self.last_shift_held = False

        self.abilities = {
            "shield": {"unlocked": False, "active": False, "duration": 0, "cooldown": 0},
            "dash": {"unlocked": False, "active": False, "duration": 0, "cooldown": 0},
        }
        self.selected_ability = None

        # Initial spawns
        for _ in range(3):
            self._spawn_specter()

        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        reward = 0.0
        self.steps += 1
        
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        space_press = space_held and not self.last_space_held
        shift_press = shift_held and not self.last_shift_held
        
        # --- UPDATE GAME LOGIC ---
        
        # 1. Handle player input and actions
        reward += self._handle_input(movement, space_press, shift_press)
        
        # 2. Update game entities
        self._update_player()
        self._update_abilities()
        self._update_specters()
        self._update_portals()
        self._update_particles()
        
        # 3. Handle spawning and progression
        self._handle_spawning()
        self._handle_progression()

        # 4. Update core state
        self.essence = max(0, self.essence - self.ESSENCE_DECAY_RATE)
        reward -= 0.01 # Small penalty for time passing to encourage action
        
        # 5. Check for termination
        terminated = False
        if self.essence <= 0:
            terminated = True
            # Sound: game over
        if self.steps >= self.MAX_STEPS:
            terminated = True
            reward += 100 # Victory reward
        
        self.game_over = terminated
        self.last_space_held, self.last_shift_held = space_held, shift_held
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    # --- PRIVATE HELPER METHODS: LOGIC ---

    def _handle_input(self, movement, space_press, shift_press):
        reward = 0.0
        
        # Movement
        move_vec = np.array([0.0, 0.0])
        if movement == 1: move_vec[1] = -1.0 # Up
        elif movement == 2: move_vec[1] = 1.0 # Down
        elif movement == 3: move_vec[0] = -1.0 # Left
        elif movement == 4: move_vec[0] = 1.0 # Right
        
        is_dashing = self.abilities["dash"]["active"]
        dash_multiplier = self.ABILITY_DASH_SPEED_MULTIPLIER if is_dashing else 1.0
        self.player_velocity = move_vec * self.PLAYER_SPEED * dash_multiplier
        
        # Ability Cycling (Shift)
        if shift_press:
            unlocked_abilities = [k for k, v in self.abilities.items() if v["unlocked"]]
            if unlocked_abilities:
                if self.selected_ability is None:
                    self.selected_ability = unlocked_abilities[0]
                else:
                    try:
                        current_index = unlocked_abilities.index(self.selected_ability)
                        next_index = (current_index + 1) % len(unlocked_abilities)
                        self.selected_ability = unlocked_abilities[next_index]
                    except ValueError:
                        self.selected_ability = unlocked_abilities[0]
                # Sound: ability cycle

        # Main Action (Space)
        if space_press:
            # 1. Try to absorb a specter
            absorbed = self._try_absorb_specter()
            if absorbed:
                reward += absorbed['reward']
            # 2. If no absorption, try to use an ability
            elif self.selected_ability:
                ability_reward = self._try_use_ability()
                reward += ability_reward
        
        return reward

    def _try_absorb_specter(self):
        closest_specter, min_dist = None, float('inf')
        for specter in self.specters:
            dist = np.linalg.norm(self.player_pos - specter['pos'])
            if dist < min_dist:
                min_dist = dist
                closest_specter = specter
        
        if closest_specter and min_dist <= self.ABSORB_RADIUS:
            is_match = (self.player_color_index == -1 or 
                        self.player_color_index == closest_specter['color_index'])
            
            if self.abilities['shield']['active']:
                is_match = True # Shield makes all absorptions successful

            if is_match:
                # Sound: positive absorb
                self.essence = min(self.MAX_ESSENCE, self.essence + self.ESSENCE_FROM_ABSORB)
                self.player_color_index = closest_specter['color_index']
                self._spawn_particles(self.player_pos, self.SPECTER_COLORS[closest_specter['color_index']], 20)
                reward = 10.0
            else:
                # Sound: negative absorb
                self.essence += self.ESSENCE_PENALTY_MISMATCH
                self.player_color_index = -1 # Reset to neutral on mismatch
                self._spawn_particles(self.player_pos, (100, 100, 100), 30, speed_mult=1.5)
                reward = -5.0

            self.specters.remove(closest_specter)
            self.score += reward
            return {'reward': reward}
        return None

    def _try_use_ability(self):
        if self.selected_ability == "shield" and self.abilities["shield"]["cooldown"] == 0:
            if self.essence >= self.ABILITY_SHIELD_COST:
                # Sound: shield activate
                self.essence -= self.ABILITY_SHIELD_COST
                self.abilities["shield"]["active"] = True
                self.abilities["shield"]["duration"] = self.ABILITY_SHIELD_DURATION
                self.abilities["shield"]["cooldown"] = self.ABILITY_SHIELD_COOLDOWN
                return -1.0 # Small cost for using ability
        elif self.selected_ability == "dash" and self.abilities["dash"]["cooldown"] == 0:
            if self.essence >= self.ABILITY_DASH_COST:
                # Sound: dash activate
                self.essence -= self.ABILITY_DASH_COST
                self.abilities["dash"]["active"] = True
                self.abilities["dash"]["duration"] = self.ABILITY_DASH_DURATION
                self.abilities["dash"]["cooldown"] = self.ABILITY_DASH_COOLDOWN
                return -1.0
        return 0.0

    def _update_player(self):
        self.player_pos += self.player_velocity
        # Toroidal world wrap
        self.player_pos[0] %= self.SCREEN_WIDTH
        self.player_pos[1] %= self.SCREEN_HEIGHT

    def _update_abilities(self):
        # Unlocks
        if not self.abilities["shield"]["unlocked"] and self.essence >= self.ABILITY_UNLOCK_SHIELD:
            self.abilities["shield"]["unlocked"] = True
        if not self.abilities["dash"]["unlocked"] and self.essence >= self.ABILITY_UNLOCK_DASH:
            self.abilities["dash"]["unlocked"] = True

        # Durations and Cooldowns
        for ability in self.abilities.values():
            if ability["active"]:
                ability["duration"] -= 1
                if ability["duration"] <= 0:
                    ability["active"] = False
            if ability["cooldown"] > 0:
                ability["cooldown"] -= 1

    def _update_specters(self):
        for specter in self.specters[:]:
            direction = self.player_pos - specter['pos']
            dist = np.linalg.norm(direction)
            if dist > 1: # Avoid division by zero
                direction /= dist
            
            specter['pos'] += direction * self.specter_speed
            specter['timer'] -= 1
            if specter['timer'] <= 0:
                self.specters.remove(specter)

    def _update_portals(self):
        for portal in self.portals[:]:
            portal['angle'] = (portal['angle'] + 2) % 360
            portal['timer'] -= 1
            dist_to_player = np.linalg.norm(self.player_pos - portal['pos'])
            
            if dist_to_player < self.PORTAL_RADIUS:
                # Sound: portal activate
                self.essence = min(self.MAX_ESSENCE, self.essence + self.PORTAL_ESSENCE_BONUS)
                reward = 20.0
                self.score += reward
                self._spawn_particles(portal['pos'], self.PORTAL_COLORS[portal['color_index']], 50, speed_mult=2.0)
                self.portals.remove(portal)
            elif portal['timer'] <= 0:
                self.portals.remove(portal)

    def _update_particles(self):
        for p in self.particles[:]:
            p['pos'] += p['vel']
            p['lifespan'] -= 1
            if p['lifespan'] <= 0:
                self.particles.remove(p)

    def _handle_spawning(self):
        # Specters
        self.specter_spawn_timer -= 1
        if self.specter_spawn_timer <= 0 and len(self.specters) < self.MAX_SPECTERS:
            self._spawn_specter()
            self.specter_spawn_timer = int(self.specter_spawn_rate)
        
        # Portals
        self.portal_spawn_timer -= 1
        if self.portal_spawn_timer <= 0 and not self.portals:
            self._spawn_portal()
            self.portal_spawn_timer = self.PORTAL_SPAWN_TIME

    def _handle_progression(self):
        if self.steps > 0 and self.steps % 200 == 0:
            self.specter_spawn_rate = max(10, self.specter_spawn_rate * 0.95)
        if self.steps > 0 and self.steps % 100 == 0:
            self.specter_speed = min(3.0, self.specter_speed + 0.05)

    def _spawn_specter(self):
        edge = self.np_random.integers(4)
        if edge == 0: pos = [self.np_random.uniform(0, self.SCREEN_WIDTH), -self.SPECTER_RADIUS]
        elif edge == 1: pos = [self.np_random.uniform(0, self.SCREEN_WIDTH), self.SCREEN_HEIGHT + self.SPECTER_RADIUS]
        elif edge == 2: pos = [-self.SPECTER_RADIUS, self.np_random.uniform(0, self.SCREEN_HEIGHT)]
        else: pos = [self.SCREEN_WIDTH + self.SPECTER_RADIUS, self.np_random.uniform(0, self.SCREEN_HEIGHT)]
        
        color_index = self.np_random.integers(len(self.SPECTER_COLORS))
        self.specters.append({
            'pos': np.array(pos, dtype=float),
            'color_index': color_index,
            'timer': self.np_random.integers(300, 600)
        })

    def _spawn_portal(self):
        pos = self.np_random.uniform(
            [self.SCREEN_WIDTH * 0.2, self.SCREEN_HEIGHT * 0.2],
            [self.SCREEN_WIDTH * 0.8, self.SCREEN_HEIGHT * 0.8]
        )
        self.portals.append({
            'pos': np.array(pos, dtype=float),
            'color_index': self.np_random.integers(len(self.PORTAL_COLORS)),
            'angle': 0,
            'timer': self.PORTAL_LIFESPAN
        })

    def _spawn_particles(self, pos, color, count, speed_mult=1.0):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 3) * speed_mult
            vel = np.array([math.cos(angle), math.sin(angle)]) * speed
            self.particles.append({
                'pos': pos.copy(),
                'vel': vel,
                'color': color,
                'lifespan': self.np_random.integers(20, 40)
            })

    # --- PRIVATE HELPER METHODS: RENDERING ---

    def _get_observation(self):
        self._draw_background()
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _draw_background(self):
        rect = pygame.Rect(0, 0, self.SCREEN_WIDTH, self.SCREEN_HEIGHT)
        pygame.draw.rect(self.screen, self.COLOR_BG_TOP, rect)
        for y in range(self.SCREEN_HEIGHT):
            alpha = int(255 * (y / self.SCREEN_HEIGHT))
            color = self.COLOR_BG_BOTTOM
            s = pygame.Surface((self.SCREEN_WIDTH, 1), pygame.SRCALPHA)
            s.fill((color[0], color[1], color[2], alpha))
            self.screen.blit(s, (0, y))

    def _render_game(self):
        self._render_portals()
        self._render_specters()
        self._render_player()
        self._render_particles()

    def _render_portals(self):
        for portal in self.portals:
            color = self.PORTAL_COLORS[portal['color_index']]
            pos = portal['pos'].astype(int)
            radius = int(self.PORTAL_RADIUS * (math.sin(self.steps * 0.1) * 0.1 + 0.95))
            for i in range(5):
                offset_angle = portal['angle'] + i * (360 / 5)
                x = pos[0] + math.cos(math.radians(offset_angle)) * radius
                y = pos[1] + math.sin(math.radians(offset_angle)) * radius
                pygame.gfxdraw.filled_circle(self.screen, int(x), int(y), 5, color)
                pygame.gfxdraw.aacircle(self.screen, int(x), int(y), 5, color)

    def _render_specters(self):
        for specter in self.specters:
            pos = specter['pos'].astype(int)
            color = self.SPECTER_COLORS[specter['color_index']]
            pulse = int(2 * math.sin(self.steps * 0.2 + pos[0]))
            radius = self.SPECTER_RADIUS + pulse
            self._draw_glow_circle(pos, radius, color)

    def _render_player(self):
        pos = self.player_pos.astype(int)
        color = self.SPECTER_COLORS[self.player_color_index] if self.player_color_index != -1 else self.COLOR_PLAYER_NEUTRAL
        
        # Player body
        self._draw_glow_circle(pos, self.PLAYER_RADIUS, color)

        # Shield effect
        if self.abilities['shield']['active']:
            alpha = 100 + 50 * math.sin(self.steps * 0.3)
            self.COLOR_SHIELD = self.COLOR_SHIELD[:3] + (int(alpha),)
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], self.PLAYER_RADIUS + 10, self.COLOR_SHIELD)
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], self.PLAYER_RADIUS + 10, self.COLOR_SHIELD)

    def _render_particles(self):
        for p in self.particles:
            pos = p['pos'].astype(int)
            alpha = int(255 * (p['lifespan'] / 40.0))
            color = p['color'] + (alpha,)
            size = int(max(1, 3 * (p['lifespan'] / 40.0)))
            pygame.draw.circle(self.screen, color, pos, size)

    def _render_ui(self):
        # Essence Bar
        bar_width = 300
        bar_height = 20
        bar_x = (self.SCREEN_WIDTH - bar_width) // 2
        bar_y = 15
        fill_ratio = self.essence / self.MAX_ESSENCE
        pygame.draw.rect(self.screen, self.COLOR_UI_BAR_BG, (bar_x, bar_y, bar_width, bar_height), border_radius=5)
        pygame.draw.rect(self.screen, self.COLOR_UI_BAR, (bar_x, bar_y, int(bar_width * fill_ratio), bar_height), border_radius=5)
        essence_text = self.font_small.render("ESSENCE", True, self.COLOR_UI_TEXT)
        self.screen.blit(essence_text, (bar_x - essence_text.get_width() - 10, bar_y))

        # Score and Steps
        score_text = self.font_small.render(f"SCORE: {int(self.score)}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (self.SCREEN_WIDTH - score_text.get_width() - 10, 10))
        steps_text = self.font_small.render(f"TIME: {self.MAX_STEPS - self.steps}", True, self.COLOR_UI_TEXT)
        self.screen.blit(steps_text, (self.SCREEN_WIDTH - steps_text.get_width() - 10, 30))

        # Ability Icons
        self._render_ability_icons()

    def _render_ability_icons(self):
        icon_y = self.SCREEN_HEIGHT - 40
        
        # Shield Icon
        if self.abilities['shield']['unlocked']:
            shield_x = self.SCREEN_WIDTH // 2 - 30
            color = self.COLOR_UI_TEXT if self.abilities['shield']['cooldown'] == 0 else (100,100,120)
            if self.selected_ability == 'shield':
                pygame.draw.rect(self.screen, self.COLOR_UI_BAR, (shield_x - 5, icon_y - 5, 30, 30), 2, border_radius=5)
            pygame.draw.rect(self.screen, color, (shield_x, icon_y, 20, 20), 2, border_radius=3)
            shield_text = self.font_small.render("SHIELD", True, color)
            self.screen.blit(shield_text, (shield_x - 4, icon_y + 22))

        # Dash Icon
        if self.abilities['dash']['unlocked']:
            dash_x = self.SCREEN_WIDTH // 2 + 30
            color = self.COLOR_UI_TEXT if self.abilities['dash']['cooldown'] == 0 else (100,100,120)
            if self.selected_ability == 'dash':
                pygame.draw.rect(self.screen, self.COLOR_UI_BAR, (dash_x - 5, icon_y - 5, 30, 30), 2, border_radius=5)
            pygame.draw.polygon(self.screen, color, [(dash_x, icon_y), (dash_x + 20, icon_y + 10), (dash_x, icon_y + 20)], 2)
            dash_text = self.font_small.render("DASH", True, color)
            self.screen.blit(dash_text, (dash_x - 2, icon_y + 22))

    def _draw_glow_circle(self, pos, radius, color):
        """Draws a circle with a glowing effect."""
        surface = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
        for i in range(radius, 0, -2):
            alpha = int(255 * (1 - (i / radius))**2)
            glow_color = color + (alpha,)
            pygame.gfxdraw.filled_circle(surface, radius, radius, i, glow_color)
        
        pygame.gfxdraw.filled_circle(surface, radius, radius, int(radius * 0.6), color)
        pygame.gfxdraw.aacircle(surface, radius, radius, int(radius * 0.6), color)
        self.screen.blit(surface, (pos[0] - radius, pos[1] - radius))


    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "essence": self.essence,
            "player_color": self.player_color_index,
            "abilities": self.abilities
        }

    def close(self):
        pygame.quit()


if __name__ == '__main__':
    # This block allows you to play the game manually
    # It will not run in a headless environment
    try:
        os.environ["SDL_VIDEODRIVER"] = "x11"
        env = GameEnv(render_mode="human")
        obs, info = env.reset()
        
        screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
        pygame.display.set_caption("Spectral Survivor")
        clock = pygame.time.Clock()
        
        terminated = False
        total_reward = 0
        
        while not terminated:
            # --- Human Controls ---
            movement = 0 # none
            keys = pygame.key.get_pressed()
            if keys[pygame.K_UP] or keys[pygame.K_w]: movement = 1
            elif keys[pygame.K_DOWN] or keys[pygame.K_s]: movement = 2
            elif keys[pygame.K_LEFT] or keys[pygame.K_a]: movement = 3
            elif keys[pygame.K_RIGHT] or keys[pygame.K_d]: movement = 4
            
            space_held = 1 if keys[pygame.K_SPACE] else 0
            shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0

            action = [movement, space_held, shift_held]

            # --- Gym Step ---
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward

            # --- Rendering ---
            surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
            screen.blit(surf, (0, 0))
            pygame.display.flip()

            # --- Event Handling & Clock ---
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    terminated = True

            clock.tick(30) # Run at 30 FPS

        print(f"Game Over! Final Score: {total_reward:.2f}, Steps: {info['steps']}")
        env.close()
    except pygame.error as e:
        print(f"Could not run in human mode (did you set a display?): {e}")
        print("This is normal in a headless environment.")